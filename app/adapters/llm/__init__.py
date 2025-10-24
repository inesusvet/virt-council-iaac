"""LLM provider adapters for AI interactions."""
import json
from typing import Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel

from app.domain.entities import Project, KnowledgeEntry
from app.domain.value_objects import MessageClassification, ResearchSuggestion


class PydanticAILLMProvider:
    """LLM provider implementation using Pydantic AI."""

    def __init__(self, provider: str, api_key: str, model_name: str):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name

        # Initialize the model based on provider
        if provider == "openai":
            self.model = OpenAIModel(model_name, api_key=api_key)
        elif provider == "gemini":
            self.model = GeminiModel(model_name, api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def classify_message(
        self, content: str, projects: list[Project]
    ) -> MessageClassification:
        """Classify a message and suggest project association."""
        # Create project context for the agent
        project_context = "\n".join(
            [f"- {p.name}: {p.description} (ID: {p.id})" for p in projects]
        )

        prompt = f"""Analyze the following message and classify it based on the available projects.

Message: {content}

Available Projects:
{project_context}

Provide:
1. A category (e.g., "feature_request", "bug_report", "question", "research", "general")
2. Confidence score (0.0 to 1.0)
3. Suggested project ID if applicable
4. Relevant tags (list of keywords)
5. Brief summary

Respond in JSON format with keys: category, confidence, suggested_project_id, tags, summary"""

        # Create an agent for classification
        agent = Agent(self.model, result_type=str)

        try:
            result = await agent.run(prompt)
            response_data = json.loads(result.data)

            return MessageClassification(
                category=response_data.get("category", "general"),
                confidence=float(response_data.get("confidence", 0.5)),
                suggested_project_id=response_data.get("suggested_project_id"),
                tags=response_data.get("tags", []),
                summary=response_data.get("summary", ""),
            )
        except Exception as e:
            # Fallback classification
            return MessageClassification(
                category="general",
                confidence=0.3,
                summary=f"Failed to classify: {str(e)}",
            )

    async def extract_knowledge(self, content: str) -> str:
        """Extract structured knowledge from message content."""
        prompt = f"""Extract key information, insights, and actionable items from this message:

Message: {content}

Provide a structured summary highlighting:
- Main topics discussed
- Key decisions or insights
- Action items or next steps
- Important context or references

Keep it concise and well-organized."""

        agent = Agent(self.model, result_type=str)

        try:
            result = await agent.run(prompt)
            return result.data
        except Exception as e:
            return f"Original message: {content}\n\nNote: Failed to extract structured knowledge: {str(e)}"

    async def suggest_next_steps(
        self, project: Project, knowledge_entries: list[KnowledgeEntry]
    ) -> list[ResearchSuggestion]:
        """Suggest next research steps based on project context."""
        # Create context from knowledge entries
        knowledge_context = "\n\n".join(
            [f"- {entry.content}" for entry in knowledge_entries[:10]]  # Limit to recent
        )

        prompt = f"""Based on the project information and knowledge base, suggest 3-5 next research steps or actions.

Project: {project.name}
Description: {project.description}

Recent Knowledge Base Entries:
{knowledge_context}

Provide suggestions in JSON array format with each item having:
- title: Brief title
- description: Detailed description
- priority: Integer (1-5, higher is more important)
- resources: List of suggested resources/tools

Example format:
[
  {{"title": "...", "description": "...", "priority": 4, "resources": ["...", "..."]}},
  ...
]"""

        agent = Agent(self.model, result_type=str)

        try:
            result = await agent.run(prompt)
            suggestions_data = json.loads(result.data)

            return [
                ResearchSuggestion(
                    title=s.get("title", ""),
                    description=s.get("description", ""),
                    priority=int(s.get("priority", 3)),
                    resources=s.get("resources", []),
                )
                for s in suggestions_data
            ]
        except Exception as e:
            # Return a default suggestion on error
            return [
                ResearchSuggestion(
                    title="Review Project Status",
                    description=f"Review the current status and next steps for {project.name}. (Error occurred: {str(e)})",
                    priority=3,
                    resources=[],
                )
            ]
