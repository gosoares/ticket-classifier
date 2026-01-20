"""LLM classification module using OpenRouter."""

import os

from dotenv import load_dotenv
from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from classifier.config import LLM_MODEL
from classifier.prompts import build_system_prompt, build_user_prompt

load_dotenv()

RETRY_PROMPT = """Sua resposta anterior não está no formato JSON válido.
Por favor, responda APENAS com JSON válido no formato:
{"classe": "<categoria>", "justificativa": "<explicação>"}"""


class ClassificationError(Exception):
    """Raised when classification fails after retries."""

    def __init__(self, ticket: str, reason: str, raw_response: str | None = None):
        self.ticket = ticket
        self.reason = reason
        self.raw_response = raw_response
        super().__init__(f"Classification failed: {reason}")


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
        max_retries: int = 1,
    ) -> ClassificationResult:
        """
        Classify a ticket using LLM.

        Args:
            ticket: The ticket text to classify
            similar_tickets: List of similar tickets from retriever
            classes: List of valid class names
            reference_tickets: Dict of representative tickets per class (optional)
            max_retries: Number of retries if JSON parsing fails

        Returns:
            ClassificationResult with classe and justificativa

        Raises:
            ClassificationError: If classification fails after all retries
        """
        system = build_system_prompt(classes)
        user = build_user_prompt(ticket, similar_tickets, reference_tickets)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_content = ""
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except APIError as e:
                raise ClassificationError(
                    ticket=ticket[:100],
                    reason=f"API error: {e.message}",
                    raw_response=None,
                ) from e

            last_content = response.choices[0].message.content or ""

            try:
                return ClassificationResult.model_validate_json(last_content)
            except ValidationError:
                if attempt < max_retries:
                    # Add assistant response and retry prompt to continue conversation
                    messages.append({"role": "assistant", "content": last_content})
                    messages.append({"role": "user", "content": RETRY_PROMPT})

        # All retries exhausted
        raise ClassificationError(
            ticket=ticket[:100],
            reason="Invalid JSON after retries",
            raw_response=last_content,
        )
