"""Classifier candidates and shared utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from classifier.config import EMBEDDING_MODEL, K_SIMILAR, RANDOM_STATE
from classifier.rag import TicketRetriever


class Classifier(Protocol):
    """Common interface for candidate classifiers."""

    name: str

    def fit(self, train_df: pd.DataFrame) -> None: ...

    def predict(self, texts: list[str]) -> list[str]: ...


class TfidfClassifier:
    """TF-IDF + linear classifier."""

    def __init__(self, classifier) -> None:
        self.classifier = classifier
        self.name = f"TF-IDF + {classifier.__class__.__name__}"
        self.model: Pipeline | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        features = self._build_tfidf_features()
        model = Pipeline(
            [
                ("features", features),
                ("classifier", self.classifier),
            ]
        )
        model.fit(train_df["Document"].tolist(), train_df["Topic_group"].tolist())
        self.model = model

    def predict(self, texts: list[str]) -> list[str]:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(texts).tolist()

    def _build_tfidf_features(self) -> FeatureUnion:
        """Create a combined word + character TF-IDF feature extractor."""
        word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=120_000,
            strip_accents="unicode",
        )
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60_000,
            strip_accents="unicode",
        )
        return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    @classmethod
    def linear_svc(
        cls, random_state: int = RANDOM_STATE, verbose: int = 0
    ) -> "TfidfClassifier":
        return cls(
            LinearSVC(
                class_weight="balanced", random_state=random_state, verbose=verbose
            )
        )

    @classmethod
    def logistic_regression(
        cls, random_state: int = RANDOM_STATE, verbose: int = 0
    ) -> "TfidfClassifier":
        return cls(
            LogisticRegression(
                max_iter=1000,
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
                verbose=verbose,
            )
        )

    @classmethod
    def sgd_classifier(
        cls, random_state: int = RANDOM_STATE, verbose: int = 0
    ) -> "TfidfClassifier":
        return cls(
            SGDClassifier(
                loss="log_loss",
                max_iter=2000,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
                verbose=verbose,
            )
        )


class EmbeddingClassifier:
    """Sentence embeddings + linear classifier."""

    def __init__(self, classifier) -> None:
        self.classifier = classifier
        self.name = f"Embeddings + {classifier.__class__.__name__}"
        self.embedding_model: SentenceTransformer | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        train_embeddings = self._embed_texts(
            train_df["Document"].tolist(),
            model=self.embedding_model,
        )
        self.classifier.fit(train_embeddings, train_df["Topic_group"].tolist())

    def predict(self, texts: list[str]) -> list[str]:
        if self.embedding_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        embeddings = self._embed_texts(texts, model=self.embedding_model)
        return self.classifier.predict(embeddings).tolist()

    def _embed_texts(
        self,
        texts: list[str],
        model: SentenceTransformer | None = None,
        model_name: str = EMBEDDING_MODEL,
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute sentence embeddings for a list of texts."""
        if model is None:
            model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)
        return embeddings

    @classmethod
    def linear_svc(
        cls, random_state: int = RANDOM_STATE, verbose: int = 0
    ) -> "EmbeddingClassifier":
        return cls(
            LinearSVC(
                class_weight="balanced", random_state=random_state, verbose=verbose
            )
        )

    @classmethod
    def logistic_regression(
        cls, random_state: int = RANDOM_STATE, verbose: int = 0
    ) -> "EmbeddingClassifier":
        return cls(
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
                verbose=verbose,
            )
        )

    @classmethod
    def sgd_classifier(
        cls, random_state: int = RANDOM_STATE, verbose: int = 0
    ) -> "EmbeddingClassifier":
        return cls(
            SGDClassifier(
                loss="log_loss",
                max_iter=2000,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
                verbose=verbose,
            )
        )


class RagKnnClassifier:
    """RAG retrieval + kNN vote classifier."""

    def __init__(self, k_similar: int = K_SIMILAR) -> None:
        self.k_similar = k_similar
        self.name = f"RAG + kNN (k={k_similar})"
        self.retriever: TicketRetriever | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.retriever = TicketRetriever()
        self.retriever.index(train_df)

    def predict(self, texts: list[str]) -> list[str]:
        if self.retriever is None:
            raise ValueError("Retriever not fitted. Call fit() first.")
        preds = []
        for text in texts:
            similar = self.retriever.retrieve(text, k=self.k_similar)
            preds.append(self._knn_predict(similar))
        return preds

    @staticmethod
    def _knn_predict(similar_tickets: list[dict]) -> str:
        if not similar_tickets:
            raise ValueError("No similar tickets to vote on.")

        class_counts: dict[str, int] = defaultdict(int)
        class_scores: dict[str, float] = defaultdict(float)
        best_scores: dict[str, float] = {}

        for ticket in similar_tickets:
            class_name = ticket["class"]
            score = float(ticket["score"])
            class_counts[class_name] += 1
            class_scores[class_name] += score
            if score > best_scores.get(class_name, float("-inf")):
                best_scores[class_name] = score

        return max(
            class_counts,
            key=lambda cls: (
                class_counts[cls],
                class_scores.get(cls, float("-inf")),
                best_scores.get(cls, float("-inf")),
            ),
        )


class RagWeightedVoteClassifier:
    """RAG retrieval + weighted vote classifier."""

    def __init__(self, k_similar: int = K_SIMILAR) -> None:
        self.k_similar = k_similar
        self.name = f"RAG + Weighted Vote (k={k_similar})"
        self.retriever: TicketRetriever | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.retriever = TicketRetriever()
        self.retriever.index(train_df)

    def predict(self, texts: list[str]) -> list[str]:
        if self.retriever is None:
            raise ValueError("Retriever not fitted. Call fit() first.")
        preds = []
        for text in texts:
            vote = self.retriever.classify_weighted_vote(text, k=self.k_similar)
            preds.append(vote["predicted_class"])
        return preds
