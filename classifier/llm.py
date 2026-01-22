"""LLM client and prompts for justification and analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI

from classifier.logging_config import get_logger
from classifier.schemas import TokenUsage

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL")

logger = get_logger("llm")


def build_system_prompt() -> str:
    """
    Gera o prompt do sistema para justificativa.

    Returns:
        Prompt formatado para o role "system"
    """
    return """Você é um assistente que gera justificativas para classificações de tickets de suporte de TI.

Responda APENAS com JSON no formato:
{"justificativa": "<explicação curta e objetiva de 1-3 frases>"}

A justificativa deve ser escrita em **Português (Brasil)**.
Use evidências do ticket e dos exemplos fornecidos para sustentar a classe informada.
Não mencione tickets similares explicitamente (ex.: “ticket 2”).

Regras:
- A justificativa deve ser auto contida.
- Não mencione tickets similares explicitamente (ex.: “ticket 2”).
- Use apenas evidências gerais do conteúdo do ticket.
- Cite explicitamente de 2 a 4 termos literais do ticket entre aspas simples (ex.: 'vpn', 'password', 'oracle').

Responda APENAS com JSON no formato:
{{"justificativa": "<explicação curta e objetiva de 1-3 frases>"}}"""


def _format_similar_tickets(tickets: list[dict]) -> str:
    """
    Formata tickets similares para o prompt.

    Args:
        tickets: Lista de dicts com 'text', 'class', 'score'

    Returns:
        String formatada com tickets numerados
    """
    if not tickets:
        return "(nenhum ticket similar encontrado)"

    lines = []
    for i, t in enumerate(tickets, 1):
        lines.append(f"{i}. [{t['class']}] {t['text']}")

    return "\n".join(lines)


def build_user_prompt(
    ticket: str,
    predicted_class: str,
    similar_tickets: list[dict],
) -> str:
    """
    Gera o prompt do usuário para justificativa.

    Args:
        ticket: Texto do ticket a classificar
        predicted_class: Classe já definida pelo classificador
        similar_tickets: Lista de tickets similares do RAG

    Returns:
        Prompt formatado para o role "user"
    """
    prompt = f"""Gere uma justificativa para a classificação abaixo.

**Classe atribuída:** {predicted_class}

**Ticket:**
{ticket}

## Tickets Similares (como evidência)
{_format_similar_tickets(similar_tickets)}"""

    return prompt


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """Result of a single LLM call."""

    content: str
    reasoning: str | None
    token_usage: TokenUsage


class LlmError(Exception):
    """Raised when an LLM request fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"LLM call failed: {reason}")


class ConclusionError(Exception):
    """Raised when conclusion generation fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Conclusion generation failed: {reason}")


class LlmClient:
    """Wrapper around OpenAI-compatible APIs."""

    model: str
    seed: int | None

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
    ):
        resolved_model = model or LLM_MODEL
        if not resolved_model:
            raise ValueError("LLM_MODEL must be set via parameter or LLM_MODEL env var")
        self.model = resolved_model
        self.seed = seed
        self.client = OpenAI(
            base_url=base_url or os.environ.get("LLM_BASE_URL"),
            api_key=api_key or os.environ.get("LLM_API_KEY"),
            timeout=20.0,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> LlmResponse:
        """
        Execute a chat completion call and return the raw response.

        Args:
            messages: Chat messages payload
            response_format: Optional response format (e.g., {"type": "json_object"})
            reasoning_effort: Reasoning effort level (low/medium/high) or None

        Returns:
            LlmResponse with content, reasoning, and token usage

        Raises:
            LlmError: If the API call fails
        """
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if response_format:
            request_kwargs["response_format"] = response_format
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
            logger.debug(f"Using seed for deterministic output: {self.seed}")
        if reasoning_effort:
            is_mimo = self.model and "mimo" in self.model.lower()
            if is_mimo:
                request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
                logger.debug("Using MiMo reasoning format (enabled)")
            else:
                request_kwargs["extra_body"] = {
                    "reasoning": {"effort": reasoning_effort}
                }
                logger.debug(
                    f"Using standard reasoning format (effort={reasoning_effort})"
                )

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except APITimeoutError as e:
            logger.error("API timeout - LLM took too long to respond")
            raise LlmError("API timeout - LLM took too long to respond") from e
        except APIError as e:
            logger.error(f"API error: {e.message}")
            raise LlmError(f"API error: {e.message}") from e

        token_usage = TokenUsage(0, 0, 0)
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        message = response.choices[0].message if response.choices else None
        content = message.content if message and message.content else ""

        reasoning = None
        reasoning_details = getattr(message, "reasoning_details", None)
        if reasoning_details:
            reasoning_texts = [
                rd.get("text", "") if isinstance(rd, dict) else getattr(rd, "text", "")
                for rd in reasoning_details
            ]
            reasoning = "\n".join(reasoning_texts) if reasoning_texts else None

        if not reasoning:
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                reasoning = reasoning_content

        return LlmResponse(
            content=content or "",
            reasoning=reasoning,
            token_usage=token_usage,
        )

    def generate_conclusion(
        self,
        system_prompt: str,
        user_prompt: str,
        reasoning_effort: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """
        Generate a conclusion text using the LLM.

        Args:
            system_prompt: System prompt for the evaluator role
            user_prompt: User prompt containing the payload and tasks
            reasoning_effort: Reasoning effort level (low/medium/high) or None

        Returns:
            Tuple of (conclusion_text, token_usage)

        Raises:
            ConclusionError: If the LLM call fails or returns empty content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.chat(
                messages,
                reasoning_effort=reasoning_effort,
            )
        except LlmError as e:
            raise ConclusionError(e.reason) from e

        content = response.content.strip()
        if not content:
            raise ConclusionError("Empty response from LLM")

        return content, response.token_usage
