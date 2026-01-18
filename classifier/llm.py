"""LLM classification module using OpenRouter."""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class ClassificationResult(BaseModel):
    """Result of ticket classification."""
    classe: str
    justificativa: str


SYSTEM_PROMPT = """Você é um classificador de tickets de suporte de TI.

Classifique o ticket em UMA das seguintes categorias:
{classes}

Responda APENAS com JSON no formato:
{{"classe": "<categoria>", "justificativa": "<explicação curta de 1-2 frases>"}}

A justificativa deve mencionar palavras-chave ou padrões do ticket que justificam a classificação."""


USER_PROMPT = """Classifique o seguinte ticket:

{ticket}

Tickets similares para referência:
{examples}"""


def format_examples(similar_tickets: list[dict]) -> str:
    """Format similar tickets for the prompt."""
    lines = []
    for i, t in enumerate(similar_tickets, 1):
        lines.append(f"{i}. [{t['class']}] {t['text'][:200]}...")
    return "\n".join(lines)


class TicketClassifier:
    """Classifies tickets using LLM with RAG context."""

    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-exp:free",
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
    ) -> ClassificationResult:
        """
        Classify a ticket using LLM.

        Args:
            ticket: The ticket text to classify
            similar_tickets: List of similar tickets from retriever
            classes: List of valid class names

        Returns:
            ClassificationResult with classe and justificativa
        """
        system = SYSTEM_PROMPT.format(classes="\n".join(f"- {c}" for c in classes))
        user = USER_PROMPT.format(
            ticket=ticket,
            examples=format_examples(similar_tickets),
        )

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
