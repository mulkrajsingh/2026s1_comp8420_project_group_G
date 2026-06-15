"""Tests for structured paper recommendation summary parsing and fallbacks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.runtime import GenerationResult
from app.synthesis import (
    build_recommended_paper_cards,
    parse_paper_recommendation_summaries,
    resolve_paper_summary,
)


def _pack() -> dict:
    return {
        "query": "artificial intelligence",
        "retrieval_mode": "offline",
        "candidates": [
            {
                "paper": {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "abstract": "Alpha abstract " * 20,
                    "authors": ["Alice"],
                    "published_date": "2020-01-01",
                    "url": "https://arxiv.org/abs/p1",
                },
                "score": 0.91,
                "apa_citation": "Alice (2020). Paper One.",
            },
            {
                "paper": {
                    "paper_id": "p2",
                    "title": "Paper Two",
                    "abstract": "Beta abstract.",
                    "authors": ["Bob"],
                    "published_date": "2021-02-02",
                    "url": "https://arxiv.org/abs/p2",
                },
                "score": 0.82,
                "apa_citation": "Bob (2021). Paper Two.",
            },
        ],
        "evidence_snippets": [
            {
                "source_id": "p1",
                "title": "Paper One",
                "snippet": "Alpha snippet.",
            },
            {
                "source_id": "p2",
                "title": "Paper Two",
                "snippet": "Beta snippet.",
            },
        ],
    }


class PaperRecommendationSummaryTests(unittest.TestCase):
    def test_parse_valid_duplicate_and_invalid_source_ids(self) -> None:
        result = GenerationResult(
            text=(
                '{"paper_summaries": ['
                '{"source_id": "p1", "summary": "Grounded one."},'
                '{"source_id": "p1", "summary": "Duplicate ignored."},'
                '{"source_id": "missing", "summary": "Unsupported."}'
                "]}"
            ),
            backend="ollama",
            model="qwen3:8b",
            prompt_id="paper_recommendation",
            task="paper_recommendation",
            strategy="zero_shot",
            latency_seconds=0.1,
            error=None,
            evidence_ids_used=["p1"],
            run_metadata={},
        )
        summaries = parse_paper_recommendation_summaries(result)
        self.assertEqual(summaries["p1"], "Grounded one.")
        self.assertNotIn("p1_duplicate", summaries)
        cards = build_recommended_paper_cards(
            [
                {
                    "paper": {"paper_id": "p1", "title": "Paper One", "abstract": "Alpha."},
                    "score": 0.9,
                    "apa_citation": "Alice (2020).",
                }
            ],
            summaries,
            {
                "evidence_snippets": [{"source_id": "p1", "snippet": "Alpha snippet."}],
            },
        )
        self.assertEqual(cards[0]["summary"], "Grounded one.")

    def test_snippet_fallback_when_summary_missing_or_generation_fails(self) -> None:
        pack = _pack()
        failed = GenerationResult(
            text="",
            backend="ollama",
            model="qwen3:8b",
            prompt_id="paper_recommendation",
            task="paper_recommendation",
            strategy="zero_shot",
            latency_seconds=0.1,
            error="timeout",
            evidence_ids_used=[],
            run_metadata={},
        )
        cards = build_recommended_paper_cards(
            pack["candidates"],
            parse_paper_recommendation_summaries(failed),
            pack,
        )
        self.assertEqual(cards[0]["summary"], "Alpha snippet.")
        self.assertEqual(cards[1]["summary"], "Beta snippet.")

    def test_metadata_comes_from_retrieval_not_llm(self) -> None:
        pack = _pack()
        result = GenerationResult(
            text='{"paper_summaries": [{"source_id": "p2", "summary": "Only p2 summarized."}]}',
            backend="ollama",
            model="qwen3:8b",
            prompt_id="paper_recommendation",
            task="paper_recommendation",
            strategy="zero_shot",
            latency_seconds=0.1,
            error=None,
            evidence_ids_used=["p2"],
            run_metadata={},
        )
        cards = build_recommended_paper_cards(
            pack["candidates"],
            parse_paper_recommendation_summaries(result),
            pack,
        )
        self.assertEqual(cards[0]["title"], "Paper One")
        self.assertEqual(cards[0]["authors"], ["Alice"])
        self.assertEqual(cards[0]["year"], "2020")
        self.assertEqual(cards[0]["apa_citation"], "Alice (2020). Paper One.")
        self.assertEqual(cards[1]["summary"], "Only p2 summarized.")
        self.assertEqual(
            resolve_paper_summary("p1", {}, pack, pack["candidates"][0]["paper"]),
            "Alpha snippet.",
        )


if __name__ == "__main__":
    unittest.main()
