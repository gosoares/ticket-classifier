"""RAG (Retrieval Augmented Generation) module for ticket classification."""

import numpy as np
import pandas as pd
from collections import defaultdict
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

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """
        Search for similar tickets using a pre-computed embedding.

        Args:
            query_embedding: Normalized embedding vector
            k: Number of tickets to retrieve

        Returns:
            List of dicts with 'text', 'class', and 'score' keys
        """
        if self.embeddings is None or self.tickets is None:
            raise ValueError("Index not built. Call index() first.")

        # Cosine similarity (dot product of normalized vectors)
        scores = self.embeddings @ query_embedding

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

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve k most similar tickets.

        Args:
            query: Query text
            k: Number of tickets to retrieve

        Returns:
            List of dicts with 'text', 'class', and 'score' keys
        """
        query_emb = self.embed(query)
        return self.search(query_emb, k)

    def weighted_vote(self, similar_tickets: list[dict]) -> dict:
        """
        Classify using a weighted vote over similar tickets.

        Each class gets the sum of similarity scores from the retrieved tickets.
        The predicted class is the one with the highest total score.

        Args:
            similar_tickets: List of dicts with 'class' and 'score' keys

        Returns:
            Dict with:
                - 'predicted_class': Class with highest weighted score
                - 'class_scores': Dict of class -> summed score (sorted desc)
                - 'best_scores': Dict of class -> best individual score
        """
        if not similar_tickets:
            raise ValueError("No similar tickets to vote on.")

        class_scores: dict[str, float] = defaultdict(float)
        best_scores: dict[str, float] = {}

        for ticket in similar_tickets:
            class_name = ticket["class"]
            score = float(ticket["score"])
            class_scores[class_name] += score
            if score > best_scores.get(class_name, float("-inf")):
                best_scores[class_name] = score

        predicted_class = max(
            class_scores,
            key=lambda cls: (class_scores[cls], best_scores.get(cls, float("-inf"))),
        )

        sorted_scores = dict(
            sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        )

        return {
            "predicted_class": predicted_class,
            "class_scores": sorted_scores,
            "best_scores": best_scores,
        }

    def classify_weighted_vote(self, query: str, k: int = 5) -> dict:
        """
        Retrieve k tickets and classify using weighted vote.

        Args:
            query: Query text to classify
            k: Number of tickets to retrieve

        Returns:
            Dict with weighted vote outputs plus 'similar_tickets'
        """
        similar = self.retrieve(query, k)
        result = self.weighted_vote(similar)
        result["similar_tickets"] = similar
        return result

    def compute_class_similarity(
        self, query: str, target_class: str, k: int = 5
    ) -> dict:
        """
        Compute similarity of a query to tickets from a specific class.

        Filters the indexed dataset by target_class, then computes cosine
        similarity between the query and those tickets.

        Args:
            query: Query text to analyze
            target_class: Class to filter tickets by
            k: Number of top scores to average (default: 5)

        Returns:
            Dict with:
                - 'mean_score': Mean of top-k similarity scores
                - 'top_scores': List of top-k scores (sorted)
                - 'top_tickets': List of top-k ticket dicts with 'text', 'score'

        Raises:
            ValueError: If index not built (call index() first)
        """
        if self.embeddings is None or self.tickets is None:
            raise ValueError("Index not built. Call index() first.")

        # Embed the query
        query_emb = self.embed(query)

        # Filter by target class
        mask = (self.tickets["Topic_group"] == target_class).values
        class_embeddings = self.embeddings[mask]
        class_tickets = self.tickets[mask].reset_index(drop=True)

        if len(class_tickets) == 0:
            return {
                "mean_score": 0.0,
                "top_scores": [],
                "top_tickets": [],
            }

        # Compute cosine similarity
        scores = class_embeddings @ query_emb

        # Get top-k
        top_k_idx = np.argsort(scores)[::-1][:min(k, len(scores))]

        top_tickets = []
        for idx in top_k_idx:
            top_tickets.append({
                "text": class_tickets.iloc[idx]["Document"],
                "score": float(scores[idx]),
            })

        return {
            "mean_score": float(scores[top_k_idx].mean()) if len(top_k_idx) > 0 else 0.0,
            "top_scores": [float(scores[i]) for i in top_k_idx],
            "top_tickets": top_tickets,
        }

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
