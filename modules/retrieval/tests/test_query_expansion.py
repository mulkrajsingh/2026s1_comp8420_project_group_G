"""Tests for conservative technical query expansion."""

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


class QueryExpansionTests(unittest.TestCase):
    def test_rag_aliases_are_added_only_for_explicit_rag_phrase(self) -> None:
        expanded = expand_query(
            "retrieval augmented generation for scientific literature"
        )
        self.assertIn("RAG", expanded)
        self.assertIn("scientific papers", expanded)
        self.assertEqual(expand_query("image retrieval"), "image retrieval")

    def test_explicit_rag_query_prioritizes_rag_papers(self) -> None:
        papers = [
            {
                "paper_id": "off-topic",
                "title": "Scientific Literature Management",
                "abstract": (
                    "Scientific literature scientific literature retrieval "
                    "for author disambiguation."
                ),
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
        )
        ranker._papers = papers

        ranked = ranker.rank(
            "retrieval augmented generation for scientific literature",
            top_k=2,
        )

        self.assertEqual(ranked[0][0]["paper_id"], "rag")


if __name__ == "__main__":
    unittest.main()
