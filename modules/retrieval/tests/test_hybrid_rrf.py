"""Tests for BM25 + SPECTER2 reciprocal rank fusion ranker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.retrieval.hybrid_ranker import (  # noqa: E402
    HybridRRFRanker,
    expand_query,
)
from app.retrieval.tfidf_bm25 import BM25Retriever  # noqa: E402


class _FakeEmbeddingRetriever:
    def __init__(self, papers: list[dict]):
        self._papers = papers

    def fit(self, papers, show_progress=True):
        self._papers = list(papers)
        return self

    def retrieve(self, query: str, top_k: int = 10):
        preferred = ["semantic", "lexical", "rag", "off-topic"]
        ranked = sorted(
            self._papers,
            key=lambda p: (
                preferred.index(p["paper_id"])
                if p["paper_id"] in preferred
                else len(preferred)
            ),
        )
        return [(p, 1.0 - i * 0.1) for i, p in enumerate(ranked[:top_k])]

    @property
    def name(self):
        return "embedding_specter2_base"


class HybridRRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.papers = [
            {
                "paper_id": "lexical",
                "title": "Support vector machine kernel classification",
                "abstract": "Kernel methods for support vector machine classification.",
                "authors": [],
                "categories": ["cs.LG"],
                "published_date": "2020-01-01",
            },
            {
                "paper_id": "semantic",
                "title": "Neural representation learning survey",
                "abstract": "Deep neural networks learn useful representations.",
                "authors": [],
                "categories": ["cs.LG"],
                "published_date": "2021-01-01",
            },
            {
                "paper_id": "off-topic",
                "title": "Bayesian graphical models",
                "abstract": "Probabilistic graphical models for inference.",
                "authors": [],
                "categories": ["stat.ML"],
                "published_date": "2019-01-01",
            },
        ]

    def test_rrf_prefers_paper_ranked_high_by_both_engines(self) -> None:
        ranker = HybridRRFRanker(
            bm25_retriever=BM25Retriever().fit(self.papers),
            embedding_retriever=_FakeEmbeddingRetriever(self.papers),
        )
        ranker._papers = self.papers

        ranked = ranker.rank(
            "support vector machine kernel classification",
            top_k=2,
        )

        self.assertEqual(ranked[0][0]["paper_id"], "lexical")

    def test_explicit_rag_query_prioritizes_rag_papers(self) -> None:
        papers = [
            {
                "paper_id": "off-topic",
                "title": "Scientific Literature Management",
                "abstract": "Scientific literature retrieval for author disambiguation.",
                "authors": [],
                "categories": ["cs.AI"],
                "published_date": "2024-01-01",
            },
            {
                "paper_id": "rag",
                "title": "A Retrieval-Augmented Generation System",
                "abstract": "RAG grounds language models in retrieved papers.",
                "authors": [],
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
            },
        ]
        ranker = HybridRRFRanker(
            bm25_retriever=BM25Retriever().fit(papers),
            embedding_retriever=_FakeEmbeddingRetriever(papers),
        )
        ranker._papers = papers

        ranked = ranker.rank(
            "retrieval augmented generation for scientific literature",
            top_k=2,
        )

        self.assertEqual(ranked[0][0]["paper_id"], "rag")

    def test_expand_query_adds_rag_aliases(self) -> None:
        expanded = expand_query(
            "retrieval augmented generation for scientific literature"
        )
        self.assertIn("RAG", expanded)


if __name__ == "__main__":
    unittest.main()
