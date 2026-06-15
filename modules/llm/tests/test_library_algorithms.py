"""Tests for library-backed retrieval and EDA calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.schemas import validate_rag_evidence_pack
from lora_dataset.arxiv_rag_retrieval import build_rag_evidence_pack, rank_papers
from scripts.lora_dataset_eda import percentile


def paper(paper_id: str, title: str, abstract: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": ["Test Author"],
        "categories": ["cs.CL"],
        "published_date": "2024-01-01",
        "venue": None,
        "doi": None,
        "arxiv_id": paper_id,
        "url": f"https://arxiv.org/abs/{paper_id}",
        "source": "test",
    }


class BM25RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.papers = [
            paper("1", "Neural retrieval", "Dense retrieval for scientific papers."),
            paper("2", "Language modelling", "Autoregressive language models."),
            paper("3", "Graph learning", "Graph neural networks for molecules."),
            paper("4", "Vision systems", "Image classification and detection."),
        ]

    def test_ranks_relevant_paper_first(self) -> None:
        ranked = rank_papers("dense retrieval", self.papers)

        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], 0)
        self.assertGreater(ranked[0][1], 0)

    def test_empty_and_unmatched_queries_have_no_hits(self) -> None:
        self.assertEqual(rank_papers("", self.papers), [])
        self.assertEqual(rank_papers("unseenword", self.papers), [])

        with self.assertRaisesRegex(ValueError, "No retrieval hits"):
            build_rag_evidence_pack("unseenword", self.papers)

    def test_evidence_pack_respects_top_k_and_schema(self) -> None:
        papers = [
            paper("1", "Dense retrieval", "Scientific document search."),
            paper("2", "Sparse retrieval", "Lexical document search."),
            paper("3", "Language modelling", "Autoregressive generation."),
            paper("4", "Graph learning", "Molecular property prediction."),
            paper("5", "Vision systems", "Image classification."),
            paper("6", "Speech recognition", "Audio transcription."),
        ]

        pack = build_rag_evidence_pack("retrieval", papers, top_k=2)

        self.assertEqual(len(pack["candidates"]), 2)
        self.assertEqual(len(pack["evidence_snippets"]), 2)
        self.assertEqual(pack["candidates"][0]["evidence"], ["S1"])
        validate_rag_evidence_pack(pack)


class PercentileTests(unittest.TestCase):
    def test_nearest_quantiles(self) -> None:
        values = [1, 2, 3, 4, 5]

        self.assertEqual(percentile(values, 0.0), 1)
        self.assertEqual(percentile(values, 0.25), 2)
        self.assertEqual(percentile(values, 0.75), 4)
        self.assertEqual(percentile(values, 1.0), 5)

    def test_empty_and_single_value_inputs(self) -> None:
        self.assertEqual(percentile([], 0.95), 0)
        self.assertEqual(percentile([7], 0.95), 7)


if __name__ == "__main__":
    unittest.main()
