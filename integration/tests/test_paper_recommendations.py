"""Integration tests for structured paper recommendation chat."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from app import pipeline, service
from app.contracts import Recommendation
from app.providers.container import Providers


class PaperRecommendationRoutingTests(unittest.TestCase):
    def test_recommendation_query_uses_concise_structured_route(self) -> None:
        analysis = pipeline.analyze_query(
            "Suggest papers on artificial intelligence"
        )
        self.assertEqual(analysis.intent, "paper_recommendation")
        self.assertEqual(analysis.style, "concise")
        self.assertTrue(analysis.should_use_retrieval)

    def test_extracted_topic_is_used_for_bank_query(self) -> None:
        recommendation = Recommendation(
            "artificial intelligence",
            [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "authors": ["Alice"],
                    "year": "2020",
                    "url": "https://arxiv.org/abs/p1",
                    "score": 0.9,
                    "apa_citation": "Alice (2020). Paper One.",
                    "summary": "Grounded summary.",
                }
            ],
        )
        rec = Mock()
        rec.recommend.return_value = recommendation
        synth = Mock()
        synth.recommend_papers.return_value = recommendation.items

        analysis = pipeline.analyze_query(
            "Recommend papers about artificial intelligence"
        )
        providers = Providers()
        providers.add("recommender", rec, "live")
        providers.add("synthesizer", synth, "live")
        response = pipeline.chat_response(
            providers,
            "Recommend papers about artificial intelligence",
            query_analysis=analysis,
        )

        rec.recommend.assert_called_once_with("artificial intelligence", [])
        self.assertEqual(response["kind"], "paper_recommendations")
        self.assertEqual(len(response["recommended_papers"]), 1)
        self.assertEqual(response["recommended_papers"][0]["title"], "Paper One")

    def test_chat_topic_registers_full_corpus_and_top_five(self) -> None:
        analysis = Mock(
            intent="paper_recommendation",
            should_use_retrieval=True,
            is_paper_recommendation=True,
            style="concise",
        )
        analysis.as_dict.return_value = {
            "intent": "paper_recommendation",
            "emotion": "neutral",
            "topic_expertise": "intermediate",
            "verbosity": "concise",
            "style": "concise",
            "confidence": 1.0,
        }
        captured: dict[str, object] = {}

        def configure_topic_providers(**kwargs):
            captured.update(kwargs)
            return Providers(), Path("/tmp/poc_corpus.jsonl")

        with (
            patch("app.service._integration_root", return_value=Path("/tmp/integration")),
            patch("app.service._request_session", return_value=(Mock(), False)),
            patch("app.service.set_active"),
            patch("app.service._finish_request_session"),
            patch("app.service.pipeline.analyze_query", return_value=analysis),
            patch(
                "app.service._topic_providers",
                side_effect=configure_topic_providers,
            ),
            patch(
                "app.service.pipeline.chat_response",
                return_value={
                    "kind": "paper_recommendations",
                    "answer": "Here are five papers on artificial intelligence:",
                    "recommended_papers": [{"paper_id": "p1"}],
                },
            ),
        ):
            response = service.chat_topic(
                "Suggest papers on artificial intelligence",
                retrieval_embedding_model="all-MiniLM-L6-v2",
                retrieval_top_k=10,
            )

        self.assertEqual(captured["query"], "artificial intelligence")
        self.assertIsNone(captured["corpus_limit"])
        self.assertEqual(captured["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(captured["top_k"], 10)
        self.assertEqual(response["kind"], "paper_recommendations")

    def test_api_shape_for_message_and_recommendations(self) -> None:
        from app.api import ChatReq, chat

        with patch(
            "app.service.chat_topic",
            return_value={
                "kind": "paper_recommendations",
                "answer": "Here are five papers on AI:",
                "recommended_papers": [
                    {
                        "paper_id": "p1",
                        "title": "Paper One",
                        "authors": ["Alice"],
                        "year": "2020",
                        "url": "https://arxiv.org/abs/p1",
                        "score": 0.9,
                        "apa_citation": "Alice (2020). Paper One.",
                        "summary": "Summary.",
                    }
                ],
            },
        ):
            payload = chat(
                ChatReq(question="Suggest papers on artificial intelligence")
            )

        self.assertEqual(payload["kind"], "paper_recommendations")
        self.assertIn("answer", payload)
        self.assertEqual(len(payload["recommended_papers"]), 1)


if __name__ == "__main__":
    unittest.main()
