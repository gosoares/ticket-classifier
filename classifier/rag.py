"""RAG (Retrieval Augmented Generation) module for ticket classification."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from classifier.config import EMBEDDING_MODEL
from classifier.logging_config import get_logger

logger = get_logger("rag")


class TicketRetriever:
    """Retrieves similar tickets using semantic search."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.debug(f"Initializing TicketRetriever with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.tickets: pd.DataFrame | None = None

    def index(self, df: pd.DataFrame) -> None:
        """
        Index tickets for retrieval.

        Args:
            df: DataFrame with 'Document' and 'Topic_group' columns
        """
        logger.info(f"Indexing {len(df):,} tickets for retrieval")
        self.tickets = df.reset_index(drop=True)
        texts = df["Document"].tolist()
        self.embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        # Normalize for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        logger.info("Indexing complete")

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
            results.append(
                {
                    "text": self.tickets.iloc[idx]["Document"],
                    "class": self.tickets.iloc[idx]["Topic_group"],
                    "score": float(scores[idx]),
                }
            )

        return results

    def compute_representatives(self) -> dict[str, dict]:
        """
        Compute representative ticket for each class (closest to centroid).

        The centroid is the mean of all embeddings for a class.
        The representative is the ticket closest to this centroid.

        Returns:
            Dict mapping class name to {'text', 'class', 'score'}
        """
        if self.embeddings is None or self.tickets is None:
            raise ValueError("Index not built. Call index() first.")

        logger.info("Computing representative tickets for each class")
        representatives = {}

        for class_name in self.tickets["Topic_group"].unique():
            # Get mask for this class
            mask = (self.tickets["Topic_group"] == class_name).values
            class_embeddings = self.embeddings[mask]

            # Compute centroid (mean of embeddings, normalized)
            centroid = class_embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Find ticket closest to centroid
            scores = class_embeddings @ centroid
            best_local_idx = int(scores.argmax())

            # Map back to original dataframe index
            original_indices = np.where(mask)[0]
            original_idx = original_indices[best_local_idx]

            representatives[class_name] = {
                "text": self.tickets.iloc[original_idx]["Document"],
                "class": class_name,
                "score": float(scores[best_local_idx]),
            }

        logger.info(f"Computed representatives for {len(representatives)} classes")
        return representatives
