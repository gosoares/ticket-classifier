"""LLM classification module using OpenAI-compatible APIs."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI
from pydantic import BaseModel, ValidationError

from classifier.logging_config import get_logger
from classifier.prompts import build_system_prompt, build_user_prompt

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL")

logger = get_logger("llm")

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


@dataclass
class TokenUsage:
    """Token usage from LLM API calls."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Sum token usage from multiple calls (e.g., retries)."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class ClassificationDetails:
    """Complete details of a classification including prompts used."""

    result: ClassificationResult
    system_prompt: str
    user_prompt: str
    similar_tickets: list[dict]
    retries: int
    token_usage: TokenUsage
    reasoning: str | None = None  # LLM reasoning/thinking process


class TicketClassifier:
    """Classifies tickets using LLM with RAG context."""

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
            timeout=20.0,  # timeout to prevent hanging
        )

    def call_llm(
        self,
        ticket: str,
        system_prompt: str,
        user_prompt: str,
        similar_tickets: list[dict],
        max_retries: int = 1,
        reasoning_effort: str | None = None,
    ) -> ClassificationDetails:
        """
        Call the LLM with pre-built prompts.

        Args:
            ticket: The ticket text (for error reporting)
            system_prompt: Pre-built system prompt
            user_prompt: Pre-built user prompt
            similar_tickets: List of similar tickets (for result metadata)
            max_retries: Number of retries if JSON parsing fails
            reasoning_effort: Reasoning effort level (low/medium/high) or None to disable

        Returns:
            ClassificationDetails with result, prompts, and metadata

        Raises:
            ClassificationError: If classification fails after all retries
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_content = ""
        last_reasoning = None
        retries_used = 0
        total_usage = TokenUsage(0, 0, 0)

        for attempt in range(max_retries + 1):
            logger.debug(f"Classification attempt {attempt + 1}/{max_retries + 1}")
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,  # Deterministic output
                }
                # Add seed if specified for additional determinism
                if self.seed is not None:
                    request_kwargs["seed"] = self.seed
                    logger.debug(f"Using seed for deterministic output: {self.seed}")
                # Add reasoning parameter if specified
                # MiMo uses "enabled" while others use "effort"
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

                response = self.client.chat.completions.create(**request_kwargs)
            except APITimeoutError as e:
                logger.error("API timeout - LLM took too long to respond")
                raise ClassificationError(
                    ticket=ticket[:100],
                    reason="API timeout - LLM took too long to respond",
                    raw_response=None,
                ) from e
            except APIError as e:
                logger.error(f"API error: {e.message}")
                raise ClassificationError(
                    ticket=ticket[:100],
                    reason=f"API error: {e.message}",
                    raw_response=None,
                ) from e

            # Accumulate token usage
            if response.usage:
                call_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                total_usage = total_usage + call_usage
                logger.debug(
                    f"Tokens used: {call_usage.total_tokens} "
                    f"(prompt: {call_usage.prompt_tokens}, "
                    f"completion: {call_usage.completion_tokens})"
                )

            message = response.choices[0].message
            last_content = message.content or ""
            logger.debug(f"LLM response: {last_content[:200]}...")

            # Extract reasoning from response
            # Try multiple formats: reasoning_details (standard), reasoning_content (MiMo)
            last_reasoning = None

            # Standard OpenRouter format
            reasoning_details = getattr(message, "reasoning_details", None)
            if reasoning_details:
                reasoning_texts = [
                    rd.get("text", "")
                    if isinstance(rd, dict)
                    else getattr(rd, "text", "")
                    for rd in reasoning_details
                ]
                last_reasoning = "\n".join(reasoning_texts) if reasoning_texts else None

            # MiMo format
            if not last_reasoning:
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content:
                    last_reasoning = reasoning_content

            if last_reasoning:
                logger.debug(f"Reasoning captured: {len(last_reasoning)} chars")

            try:
                result = ClassificationResult.model_validate_json(last_content)
                logger.debug(f"Classification successful: {result.classe}")
                return ClassificationDetails(
                    result=result,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    similar_tickets=similar_tickets,
                    retries=retries_used,
                    token_usage=total_usage,
                    reasoning=last_reasoning,
                )
            except ValidationError:
                retries_used += 1
                logger.warning(f"Invalid JSON response, retry {retries_used}")
                if attempt < max_retries:
                    # Add assistant response and retry prompt to continue conversation
                    messages.append({"role": "assistant", "content": last_content})
                    messages.append({"role": "user", "content": RETRY_PROMPT})

        # All retries exhausted
        logger.error(f"Classification failed after {retries_used} retries")
        raise ClassificationError(
            ticket=ticket[:100],
            reason="Invalid JSON after retries",
            raw_response=last_content,
        )

    def classify(
        self,
        ticket: str,
        similar_tickets: list[dict],
        classes: list[str],
        reference_tickets: dict[str, dict] | None = None,
        max_retries: int = 1,
        reasoning_effort: str | None = None,
    ) -> ClassificationDetails:
        """
        Classify a ticket using LLM (convenience method that builds prompts).

        Args:
            ticket: The ticket text to classify
            similar_tickets: List of similar tickets from retriever
            classes: List of valid class names
            reference_tickets: Dict of representative tickets per class (optional)
            max_retries: Number of retries if JSON parsing fails
            reasoning_effort: Reasoning effort level (low/medium/high) or None

        Returns:
            ClassificationDetails with result, prompts, and metadata

        Raises:
            ClassificationError: If classification fails after all retries
        """
        logger.debug("Building prompts for classification")
        system = build_system_prompt(classes)
        user = build_user_prompt(ticket, similar_tickets, reference_tickets)
        return self.call_llm(
            ticket, system, user, similar_tickets, max_retries, reasoning_effort
        )
