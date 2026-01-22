"""Shared schemas and lightweight data structures."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


class JustificationResult(BaseModel):
    """Result of ticket justification."""

    justificativa: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage from LLM API calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(slots=True)
class JustificationDetails:
    """Complete details of a justification including evidence and metadata."""

    predicted_class: str
    result: JustificationResult
    similar_tickets: list[dict]
    justification_source: str
    retries: int = 0
    token_usage: TokenUsage = TokenUsage()
    system_prompt: str | None = None
    user_prompt: str | None = None
    reasoning: str | None = None
    evidence_terms: list[str] | None = None
