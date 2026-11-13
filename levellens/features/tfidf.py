"""Sparse TF-IDF features for Share D's text models."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_tfidf(*, min_df: int = 2, word_limit: int = 50_000, char_limit: int = 100_000):
    """Combine word 1–2 grams and character 3–5 grams in a single sparse matrix."""
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
            min_df=min_df, max_features=word_limit, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            min_df=min_df, max_features=char_limit, dtype=np.float32,
        )),
    ])


def make_r2de_tfidf(*, max_features: int, min_df: int):
    """Build the sparse word-unigram features for the R2DE-style forest."""
    return TfidfVectorizer(
        ngram_range=(1, 1), stop_words="english", max_features=max_features,
        min_df=min_df, dtype=np.float32,
    )
