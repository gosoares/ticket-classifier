"""RAG (Retrieval Augmented Generation) module for ticket classification."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class TicketRetriever:
    """Retrieves similar tickets using semantic search."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.tickets: pd.DataFrame | None = None

    def index(self, df: pd.DataFrame) -> None:
        """
        Index tickets for retrieval.

        Args:
            df: DataFrame with 'Document' and 'Topic_group' columns
        """
        self.tickets = df.reset_index(drop=True)
        texts = df["Document"].tolist()
        self.embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        # Normalize for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve k most similar tickets.

        Args:
            query: Query text
            k: Number of tickets to retrieve

        Returns:
            List of dicts with 'text', 'class', and 'score' keys
        """
        if self.embeddings is None or self.tickets is None:
            raise ValueError("Index not built. Call index() first.")

        # Embed and normalize query
        query_emb = self.model.encode(query, convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Cosine similarity (dot product of normalized vectors)
        scores = self.embeddings @ query_emb

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append({
                "text": self.tickets.iloc[idx]["Document"],
                "class": self.tickets.iloc[idx]["Topic_group"],
                "score": float(scores[idx]),
            })

        return results
