"""Shared feature extraction components."""

from __future__ import annotations

import re
import unicodedata
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def preprocess_text(text: str) -> str:
    """
    Normalize text to match dataset format.

    Applies:
    - Lowercase
    - Remove accents (NFD decomposition)
    - Remove punctuation
    - Normalize whitespace

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return text

    # 1. Lowercase
    text = text.lower()

    # 2. Remove accents (NFD decomposition + remove non-ASCII marks)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", punctuation))

    # 4. Normalize whitespace (collapse multiple spaces, strip)
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TfidfFeatureExtractor:
    """Reusable TF-IDF feature extractor (word n-grams by default)."""

    def __init__(
        self,
        *,
        use_char_ngrams: bool = False,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        max_word_features: int = 120_000,
        max_char_features: int = 60_000,
    ) -> None:
        word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            min_df=min_df,
            max_features=max_word_features,
            strip_accents=None,
        )
        if use_char_ngrams:
            char_tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=char_ngram_range,
                min_df=min_df,
                max_features=max_char_features,
                strip_accents=None,
            )
            self.vectorizer = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
        else:
            self.vectorizer = word_tfidf

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
