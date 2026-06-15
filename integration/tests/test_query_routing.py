"""Integration chat routing tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from app import cli, llm_bridge, pipeline, service
from app.contracts import RagEvidencePack, Recommendation
from app.providers.container import Providers
from app.providers.live_providers import SubprocessSynthesizer


class QueryRoutingTests(unittest.TestCase):
    def test_llm_bridge_loads_react_module_with_relative_imports(self) -> None:
        module = llm_bridge._react_loop_module()

        parsed = module.parse_tool_call(
            '{"tool":"search_offline","arguments":{"query":"RAG","top_k":3}}'
        )

        self.assertEqual(parsed["arguments"]["query"], "RAG")

    def test_interactive_rag_answer_uses_fast_llm_verification(self) -> None:
        synthesizer = SubprocessSynthesizer()
        with patch.object(
            synthesizer,
            "_run_synthesis_cli",
            return_value={
                "summary": "Grounded answer.",
                "key_findings": [],
                "research_gaps": [],
            },
        ) as run_synthesis:
            answer = synthesizer.answer(
                None,
                "How does RAG work?",
                RagEvidencePack("How does RAG work?", []),
            )

        self.assertEqual(answer, "Grounded answer.")
        run_synthesis.assert_called_once_with(
            synthesizer.integration_outputs / "rag_evidence_pack.json",
            fast_verify=True,
        )

    def test_paper_chat_cli_prints_pdf_chat_response(self) -> None:
        with (
            patch.object(
                service,
                "run_pdf_chat",
                return_value="grounded paper answer",
            ) as run_pdf_chat,
            patch.object(cli.typer, "echo") as echo,
        ):
            cli.chat(
                "Which baseline is used?",
                paper_json="paper.json",
                llm_model="qwen3:8b",
                style="auto",
            )

        run_pdf_chat.assert_called_once()
        echo.assert_called_once_with("grounded paper answer")

    def test_capability_greeting_is_classified_as_direct_chat(self) -> None:
        analysis = pipeline.analyze_query(
            "Hi, I am new to this tool. What can you help me with?"
        )

        self.assertEqual(analysis.intent, "chit_chat")
        self.assertFalse(analysis.should_use_retrieval)
        self.assertEqual(analysis.field_sources["intent"], "pattern_match")

    def test_chit_chat_uses_direct_synthesizer_without_retrieval(self) -> None:
        analysis = Mock(intent="chit_chat", should_use_retrieval=False, is_paper_recommendation=False)
        direct = Mock()
        direct.answer_direct.return_value = "generated greeting"
        providers = Providers().add("synthesizer", direct, "live")
        with patch.object(pipeline, "analyze_query", return_value=analysis):
            answer = pipeline.chat(providers, "Hello, how are you?")

        self.assertEqual(answer, "generated greeting")
        direct.answer_direct.assert_called_once_with("Hello, how are you?")

    def test_precomputed_analysis_is_reused(self) -> None:
        analysis = Mock(intent="chit_chat", should_use_retrieval=False, is_paper_recommendation=False)
        direct = Mock()
        direct.answer_direct.return_value = "generated greeting"
        providers = Providers().add("synthesizer", direct, "live")
        with patch.object(
            pipeline,
            "analyze_query",
            side_effect=AssertionError("query must not be classified twice"),
        ):
            answer = pipeline.chat(
                providers,
                "hi",
                query_analysis=analysis,
            )

        self.assertEqual(answer, "generated greeting")

    def test_retrieval_chat_uses_react_when_search_offline_pack_exists(self) -> None:
        analysis = Mock(
            intent="research_question",
            should_use_retrieval=True,
            is_paper_recommendation=False,
        )
        synthesizer = Mock()
        synthesizer.model = "qwen3:8b"
        synthesizer.answer.return_value = "evidence-backed answer"
        pack = {
            "query": "transformer attention",
            "retrieval_mode": "offline",
            "candidates": [
                {
                    "paper": {
                        "paper_id": "p1",
                        "title": "Attention Paper",
                        "authors": ["Alice"],
                        "published_date": "2024-01-01",
                        "url": "https://arxiv.org/abs/p1",
                    },
                    "score": 0.9,
                    "apa_citation": "Alice (2024). Attention Paper.",
                    "reason": "Relevant to transformer attention.",
                }
            ],
            "evidence_snippets": [{"source_id": "p1", "snippet": "text"}],
        }
        plan = Mock(
            used_react=True,
            retrieval_query="transformer attention",
            fallback_reason=None,
        )
        recommender = Mock()
        recommender.search_offline_pack = Mock(return_value=pack)
        providers = Providers().add("synthesizer", synthesizer, "live").add(
            "recommender",
            recommender,
            "live",
        )
        with (
            patch.object(pipeline, "analyze_query", return_value=analysis),
            patch("app.llm_bridge.build_llm_runtime", return_value=Mock()),
            patch(
                "app.llm_bridge.run_react_topic_rag",
                return_value=(pack, plan),
            ),
        ):
            result = pipeline.chat_response(
                providers, "How does transformer attention work?"
            )

        self.assertEqual(result["answer"], "evidence-backed answer")
        self.assertTrue(result.get("react_used"))
        self.assertEqual(result["recommended_papers"][0]["paper_id"], "p1")
        self.assertEqual(
            result["apa_citations"],
            ["Alice (2024). Attention Paper."],
        )
        recommender.recommend.assert_not_called()
        synthesizer.answer.assert_called_once()

    def test_retrieval_chat_fallback_returns_cached_recommendation_sources(self) -> None:
        analysis = Mock(
            intent="research_question",
            should_use_retrieval=True,
            is_paper_recommendation=False,
        )
        evidence = RagEvidencePack(
            "retrieval augmented generation",
            [{"paper_id": "p1", "text": "grounding", "score": 1.0}],
        )
        recommendation = Recommendation(
            "retrieval augmented generation",
            [
                {
                    "paper_id": "p1",
                    "title": "Grounded Generation",
                    "url": "https://arxiv.org/abs/p1",
                    "score": 0.8,
                    "apa_citation": "Alice (2024). Grounded Generation.",
                }
            ],
        )
        recommender = Mock(spec=["retrieve_evidence", "recommend"])
        recommender.retrieve_evidence.return_value = evidence
        recommender.recommend.return_value = recommendation
        synthesizer = Mock()
        synthesizer.answer.return_value = "grounded answer"
        providers = Providers().add("synthesizer", synthesizer, "live").add(
            "recommender",
            recommender,
            "live",
        )

        result = pipeline.chat_response(
            providers,
            "How does retrieval augmented generation work?",
            query_analysis=analysis,
        )

        recommender.retrieve_evidence.assert_called_once_with(
            "How does retrieval augmented generation work?",
            mode="offline",
        )
        recommender.recommend.assert_called_once_with(
            "How does retrieval augmented generation work?",
            [],
        )
        self.assertEqual(result["answer"], "grounded answer")
        self.assertEqual(result["recommended_papers"], recommendation.items)
        self.assertEqual(
            result["apa_citations"],
            ["Alice (2024). Grounded Generation."],
        )


if __name__ == "__main__":
    unittest.main()
