"""LLM classification module using OpenRouter."""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from classifier.config import LLM_MODEL
from classifier.prompts import build_system_prompt, build_user_prompt

load_dotenv()


class ClassificationResult(BaseModel):
    """Result of ticket classification."""

    classe: str
    justificativa: str


class TicketClassifier:
    """Classifies tickets using LLM with RAG context."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        api_key: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        )

    def classify(
        self,
        ticket: str,
        similar_tickets: list[dict],
        classes: list[str],
        reference_tickets: dict[str, dict] | None = None,
    ) -> ClassificationResult:
        """
        Classify a ticket using LLM.

        Args:
            ticket: The ticket text to classify
            similar_tickets: List of similar tickets from retriever
            classes: List of valid class names
            reference_tickets: Dict of representative tickets per class (optional)

        Returns:
            ClassificationResult with classe and justificativa
        """
        system = build_system_prompt(classes)
        user = build_user_prompt(ticket, similar_tickets, reference_tickets)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        return ClassificationResult.model_validate_json(content)
