"""Justification generators (LLM and non-LLM)."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from classifier.features import TfidfFeatureExtractor
from classifier.llm import LlmClient, LlmError, build_system_prompt, build_user_prompt
from classifier.rag import TicketRetriever
from classifier.schemas import JustificationDetails, JustificationResult, TokenUsage


RETRY_PROMPT = """Sua resposta anterior não está no formato JSON válido.
Por favor, responda APENAS com JSON válido no formato:
{"justificativa": "<explicação>"}"""


class JustificationError(Exception):
    """Raised when justification fails after retries."""

    def __init__(self, ticket: str, reason: str, raw_response: str | None = None):
        self.ticket = ticket
        self.reason = reason
        self.raw_response = raw_response
        super().__init__(f"Justification failed: {reason}")


class Justifier(Protocol):
    """Common interface for justification strategies."""

    name: str

    def justify(
        self,
        ticket: str,
        predicted_class: str,
        similar_tickets: list[dict] | None = None,
    ) -> JustificationDetails: ...


class LinearJustifier:
    """Deterministic justifier based on linear model feature contributions."""

    def __init__(
        self,
        coef: npt.NDArray[np.floating],
        classes: Sequence[str],
        features: TfidfFeatureExtractor,
        top_k: int = 4,
    ) -> None:
        self.coef = coef
        self.classes = classes
        self.features = features
        self.top_k = top_k
        self.name = "Linear evidence (TF-IDF weights)"
        self._feature_names: list[str] | None = None
        self._class_to_index: dict[str, int] | None = None
        self._word_feature_prefix: str | None = None

    def _infer_word_prefix(self) -> str:
        if self._feature_names is None:
            return ""
        if any(name.startswith("word__") for name in self._feature_names):
            return "word__"
        return ""

    def _top_terms(
        self,
        text: str,
        predicted_class: str,
        *,
        top_k: int,
        feature_prefix: str | None = None,
        require_positive: bool = True,
    ) -> list[str]:
        if self._feature_names is None:
            self._feature_names = list(self.features.get_feature_names_out())
        if self._word_feature_prefix is None:
            self._word_feature_prefix = self._infer_word_prefix()
        if self._class_to_index is None:
            self._class_to_index = {str(c): i for i, c in enumerate(self.classes)}
        if feature_prefix is None:
            feature_prefix = self._word_feature_prefix

        class_index = self._class_to_index.get(str(predicted_class))
        if class_index is None:
            raise ValueError(f"Unknown class '{predicted_class}' for this model.")

        x = self.features.transform([text]).tocsr()
        indices = x.indices
        data = x.data
        if len(indices) == 0:
            return []

        weights = self.coef[0] if self.coef.shape[0] == 1 else self.coef[class_index]
        contrib = data * weights[indices]
        order = np.argsort(contrib)[::-1]

        terms: list[str] = []
        for pos in order:
            idx = int(indices[pos])
            score = float(contrib[pos])
            if require_positive and score <= 0:
                break

            name = str(self._feature_names[idx])
            if feature_prefix and not name.startswith(feature_prefix):
                continue
            term = name[len(feature_prefix) :] if feature_prefix else name
            if not term:
                continue
            terms.append(term)
            if len(terms) >= top_k:
                break

        return terms

    def justify(
        self,
        ticket: str,
        predicted_class: str,
        similar_tickets: list[dict] | None = None,
    ) -> JustificationDetails:
        terms = self._top_terms(
            ticket,
            predicted_class,
            top_k=self.top_k,
            require_positive=True,
        )
        evidence_terms = terms[: self.top_k]
        quoted = (
            ", ".join(f"'{t}'" for t in evidence_terms)
            if evidence_terms
            else "termos do texto"
        )
        justification = (
            f"O ticket foi classificado como '{predicted_class}' por mencionar {quoted}, "
            "que são termos comumente associados a essa categoria."
        )

        return JustificationDetails(
            predicted_class=predicted_class,
            result=JustificationResult(justificativa=justification),
            similar_tickets=[],
            justification_source="linear",
            retries=0,
            token_usage=TokenUsage(),
            system_prompt=None,
            user_prompt=None,
            reasoning=None,
            evidence_terms=evidence_terms,
        )


class LlmJustifier:
    """LLM justifier with RAG evidence in the prompt."""

    def __init__(
        self,
        llm: LlmClient,
        retriever: TicketRetriever,
        *,
        k_similar: int = 5,
        reasoning_effort: str | None = None,
        max_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.k_similar = k_similar
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.name = f"LLM ({llm.model})"

    def justify(
        self,
        ticket: str,
        predicted_class: str,
        similar_tickets: list[dict] | None = None,
    ) -> JustificationDetails:
        similar_tickets = (
            similar_tickets
            if similar_tickets is not None
            else self.retriever.retrieve(ticket, k=self.k_similar)
        )
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(ticket, predicted_class, similar_tickets)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_content = ""
        last_reasoning = None
        retries_used = 0
        total_usage = TokenUsage(0, 0, 0)
        last_failure_reason: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.chat(
                    messages,
                    response_format={"type": "json_object"},
                    reasoning_effort=self.reasoning_effort,
                )
            except LlmError as e:
                raise JustificationError(
                    ticket=ticket[:100],
                    reason=e.reason,
                    raw_response=None,
                ) from e

            total_usage = total_usage + response.token_usage
            last_content = response.content or ""
            last_reasoning = response.reasoning

            try:
                result = JustificationResult.model_validate_json(last_content)
                return JustificationDetails(
                    predicted_class=predicted_class,
                    result=result,
                    similar_tickets=similar_tickets,
                    justification_source="llm",
                    retries=retries_used,
                    token_usage=total_usage,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    reasoning=last_reasoning,
                )
            except ValidationError:
                retries_used += 1
                last_failure_reason = "Invalid JSON or missing required fields"
                if attempt < self.max_retries:
                    messages.append({"role": "assistant", "content": last_content})
                    messages.append({"role": "user", "content": RETRY_PROMPT})
                else:
                    break

        raise JustificationError(
            ticket=ticket[:100],
            reason=last_failure_reason or "Invalid response after retries",
            raw_response=last_content,
        )
