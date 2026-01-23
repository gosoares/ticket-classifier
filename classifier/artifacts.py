"""Model and retrieval artifacts persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

from classifier.classifiers import TfidfClassifier


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata saved alongside trained artifacts."""

    schema_version: int
    random_state: int
    classes: list[str]
    model_name: str
    embedding_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_model(
    classifier: TfidfClassifier,
    metadata: ModelMetadata,
    artifacts_dir: str | Path,
) -> Path:
    """Persist trained model + metadata."""
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "model.joblib"
    joblib.dump(classifier, model_path)

    metadata_path = artifacts_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    return model_path


def load_model(artifacts_dir: str | Path) -> TfidfClassifier:
    """Load a trained model from artifacts."""
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / "model.joblib"
    return joblib.load(model_path)


def load_metadata(artifacts_dir: str | Path) -> ModelMetadata:
    """Load metadata saved alongside trained artifacts."""
    artifacts_path = Path(artifacts_dir)
    metadata_path = artifacts_path / "metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ModelMetadata(**data)


def save_rag_index(
    retriever,
    artifacts_dir: str | Path,
) -> Path:
    """Persist RAG embeddings + tickets."""
    # Local import to avoid pulling in sentence-transformers/torch unless needed.
    from classifier.rag import TicketRetriever

    if not isinstance(retriever, TicketRetriever):
        raise TypeError("retriever must be a TicketRetriever")
    rag_path = Path(artifacts_dir) / "rag"
    rag_path.mkdir(parents=True, exist_ok=True)
    retriever.save_index(rag_path)
    return rag_path


def load_rag_index(
    artifacts_dir: str | Path,
) -> "TicketRetriever":
    """Load RAG embeddings + tickets from artifacts."""
    # Local import to avoid pulling in sentence-transformers/torch unless needed.
    from classifier.rag import TicketRetriever

    rag_path = Path(artifacts_dir) / "rag"
    meta_path = rag_path / "meta.json"
    model_name = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = meta.get("embedding_model")
    return TicketRetriever.load_index(rag_path, model_name=model_name or None)
