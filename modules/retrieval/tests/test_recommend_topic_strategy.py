"""Tests for recommend-topic retrieval strategy selection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.cli import _tfidf_recommendations, cmd_recommend_topic  # noqa: E402


def _sample_papers() -> list[dict]:
    base = {
        "venue": "",
        "doi": "",
        "arxiv_id": "",
        "url": "",
        "source": "test",
    }
    return [
        {
            **base,
            "paper_id": "p1",
            "title": "Retrieval augmented generation for scientific papers",
            "abstract": "RAG combines retrieval with language model generation.",
            "authors": ["Ada Lovelace"],
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
        },
        {
            **base,
            "paper_id": "p2",
            "title": "Bayesian graphical models",
            "abstract": "Probabilistic inference with graphical models.",
            "authors": ["Bob Turing"],
            "categories": ["stat.ML"],
            "published_date": "2019-01-01",
        },
    ]


class RecommendTopicStrategyTests(unittest.TestCase):
    def test_tfidf_recommendations_shape(self) -> None:
        papers = _sample_papers()
        recommendations = _tfidf_recommendations(
            papers,
            "retrieval augmented generation",
            top_k=2,
        )

        self.assertGreaterEqual(len(recommendations), 1)
        first = recommendations[0]
        self.assertIn("paper", first)
        self.assertIn("score", first)
        self.assertIn("apa_citation", first)
        self.assertIn("reason", first)
        self.assertEqual(first["paper"]["paper_id"], "p1")

    def test_recommend_topic_tfidf_writes_json(self) -> None:
        papers = _sample_papers()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus.jsonl"
            with corpus.open("w", encoding="utf-8") as handle:
                for paper in papers:
                    handle.write(json.dumps(paper) + "\n")
            out_path = root / "recommendations.json"

            args = Namespace(
                query="retrieval augmented generation",
                papers=str(corpus),
                out=str(out_path),
                top_k=2,
                embedding_model=None,
                retrieval_strategy="tfidf",
            )
            cmd_recommend_topic(args)

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertGreaterEqual(len(payload), 1)
            self.assertEqual(payload[0]["paper"]["paper_id"], "p1")
            self.assertTrue((root / "rag_evidence_pack.json").is_file())


if __name__ == "__main__":
    unittest.main()
