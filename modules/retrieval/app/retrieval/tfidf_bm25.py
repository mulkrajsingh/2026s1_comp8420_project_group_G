"""TF-IDF and BM25 lexical retrieval baselines.

Both retrievers tokenize title-plus-abstract text with a shared sklearn analyzer
so lexical scores are directly comparable in evaluation tables.
"""

from __future__ import annotations

import numpy as np


def build_shared_analyzer():
    """Analyzer shared by TF-IDF and BM25 so both tokenize identically."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    ).build_analyzer()


class TfidfRetriever:
    """TF-IDF cosine similarity over title + abstract (sklearn baseline)."""

    def __init__(self, max_features: int = 50_000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
        )
        self._matrix = None
        self._papers: list[dict] = []

    def fit(self, papers: list[dict]) -> "TfidfRetriever":
        self._papers = list(papers)
        corpus = [f"{p['title']} {p['abstract']}" for p in self._papers]
        self._matrix = self.vectorizer.fit_transform(corpus)
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._papers[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    @property
    def name(self) -> str:
        return "tfidf"


class BM25Retriever:
    """BM25 retrieval baseline (rank-bm25 BM25L variant).

    Tokenisation is shared with :class:`TfidfRetriever` via
    :func:`build_shared_analyzer`. BM25L handles short title+abstract documents
    better than Okapi BM25 on this homogeneous ML/CS corpus.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.4):
        self._k1 = k1
        self._b = b
        self._analyzer = build_shared_analyzer()
        self._bm25 = None
        self._papers: list[dict] = []

    def fit(self, papers: list[dict]) -> "BM25Retriever":
        try:
            from rank_bm25 import BM25L as _BM25
        except ImportError:
            try:
                from rank_bm25 import BM25Okapi as _BM25
            except ImportError as exc:
                raise ImportError("Install rank-bm25: pip install rank-bm25") from exc

        self._papers = list(papers)
        corpus = [
            self._analyzer(f"{p['title']} {p['abstract']}")
            for p in self._papers
        ]
        self._bm25 = _BM25(corpus, k1=self._k1, b=self._b)
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        tokens = self._analyzer(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._papers[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    @property
    def name(self) -> str:
        return "bm25"
