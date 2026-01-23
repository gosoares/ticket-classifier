"""UI helpers + entrypoint for running the Streamlit UI via `uv run ui`.

This module provides:
- a small runtime wrapper for interactive UIs (Streamlit),
- a CLI entrypoint to launch the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from classifier.artifacts import load_metadata, load_model
from classifier.justifiers import LinearJustifier
from classifier.schemas import JustificationDetails

if TYPE_CHECKING:
    from classifier.rag import TicketRetriever
else:
    TicketRetriever = Any


@dataclass(frozen=True, slots=True)
class ArtifactsStatus:
    artifacts_dir: Path
    model_path: Path
    metadata_path: Path
    rag_dir: Path
    has_model: bool
    has_metadata: bool
    has_rag: bool


def get_artifacts_status(artifacts_dir: str | Path = "artifacts") -> ArtifactsStatus:
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / "model.joblib"
    metadata_path = artifacts_path / "metadata.json"
    rag_dir = artifacts_path / "rag"
    return ArtifactsStatus(
        artifacts_dir=artifacts_path,
        model_path=model_path,
        metadata_path=metadata_path,
        rag_dir=rag_dir,
        has_model=model_path.exists(),
        has_metadata=metadata_path.exists(),
        has_rag=rag_dir.exists(),
    )


def build_justifier(
    *,
    classifier,
    justification_mode: str,
    retriever: TicketRetriever | None,
    k_similar: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
):
    """Create a justifier consistent with the CLI behavior."""
    if justification_mode == "linear":
        return LinearJustifier(
            classifier.classifier.coef_,
            classifier.classifier.classes_,
            classifier.features,
        )

    if justification_mode != "llm":
        raise ValueError("justification_mode must be 'linear' or 'llm'")

    if retriever is None:
        raise ValueError(
            "RAG index not found. Train and save artifacts with RAG enabled."
        )

    from classifier.justifiers import LlmJustifier
    from classifier.llm import LlmClient

    llm_client = LlmClient(model=model, seed=None)
    return LlmJustifier(
        llm_client,
        retriever,
        k_similar=k_similar,
        reasoning_effort=reasoning_effort,
    )


def load_runtime(
    *,
    artifacts_dir: str | Path = "artifacts",
    justification_mode: str = "linear",
):
    """Load model + metadata, and optionally the RAG index (for LLM mode)."""
    status = get_artifacts_status(artifacts_dir)
    if not status.has_model:
        raise FileNotFoundError(f"Missing model artifact: {status.model_path}")
    if not status.has_metadata:
        raise FileNotFoundError(f"Missing metadata artifact: {status.metadata_path}")

    classifier = load_model(status.artifacts_dir)
    metadata = load_metadata(status.artifacts_dir)

    retriever = None
    if justification_mode == "llm":
        if not status.has_rag:
            raise FileNotFoundError(f"Missing RAG artifact dir: {status.rag_dir}")
        from classifier.artifacts import load_rag_index

        retriever = load_rag_index(status.artifacts_dir)

    return classifier, metadata, retriever


def classify_one(
    *,
    ticket: str,
    classifier,
    justification_mode: str,
    k_similar: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    retriever: TicketRetriever | None = None,
) -> tuple[dict, JustificationDetails]:
    """Classify one ticket and return the challenge payload + full details."""
    ticket = (ticket or "").strip()
    if not ticket:
        raise ValueError("Ticket text is required.")

    justifier = build_justifier(
        classifier=classifier,
        justification_mode=justification_mode,
        retriever=retriever,
        k_similar=k_similar,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    predicted_class = classifier.predict([ticket])[0]
    similar_tickets = []
    if justification_mode == "llm":
        if retriever is None:
            raise ValueError("Retriever not available for LLM justifications.")
        similar_tickets = retriever.retrieve(ticket, k=k_similar)

    details = justifier.justify(
        ticket=ticket,
        predicted_class=predicted_class,
        similar_tickets=similar_tickets or None,
    )
    output = {"classe": predicted_class, "justificativa": details.result.justificativa}
    return output, details


def main() -> int:
    # Import here so normal library usage doesn't require Streamlit.
    from streamlit.web import cli as stcli

    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")

    # Pass through extra Streamlit CLI args, e.g.:
    # `uv run ui -- --server.port 8502`
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    stcli.main()
    return 0
