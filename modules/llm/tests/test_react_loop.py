"""Tests for bounded ReAct topic RAG planning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.react_loop import (  # noqa: E402
    parse_tool_call,
    plan_search_offline,
    run_react_topic_rag,
)
from app.runtime import GenerationResult  # noqa: E402


class ReactLoopTests(unittest.TestCase):
    def test_parse_tool_call_accepts_fenced_json(self) -> None:
        text = (
            'Here is the plan:\n```json\n'
            '{"tool":"search_offline","arguments":{"query":"RAG papers","top_k":3}}\n```'
        )
        parsed = parse_tool_call(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["arguments"]["query"], "RAG papers")
        self.assertEqual(parsed["arguments"]["top_k"], 3)

    def test_parse_tool_call_rejects_invalid_payload(self) -> None:
        self.assertIsNone(parse_tool_call("not json"))
        self.assertIsNone(parse_tool_call('{"tool":"other","arguments":{"query":"x"}}'))

    def test_plan_search_offline_falls_back_on_invalid_json(self) -> None:
        runtime = Mock()
        runtime.generate.return_value = GenerationResult(
            text="I will search now",
            backend="ollama",
            model="qwen3:8b",
            prompt_id="react_topic_plan",
            task="react_tool_plan",
            strategy="zero_shot",
            latency_seconds=0.1,
            error=None,
            evidence_ids_used=[],
            run_metadata={},
        )
        plan = plan_search_offline(
            "What is retrieval augmented generation?",
            runtime=runtime,
            model="qwen3:8b",
        )
        self.assertFalse(plan.used_react)
        self.assertEqual(plan.retrieval_query, "What is retrieval augmented generation?")
        self.assertEqual(plan.fallback_reason, "invalid_tool_json")

    def test_run_react_topic_rag_executes_search_offline(self) -> None:
        runtime = Mock()
        runtime.generate.return_value = GenerationResult(
            text='{"tool":"search_offline","arguments":{"query":"dense retrieval","top_k":4}}',
            backend="ollama",
            model="qwen3:8b",
            prompt_id="react_topic_plan",
            task="react_tool_plan",
            strategy="zero_shot",
            latency_seconds=0.2,
            error=None,
            evidence_ids_used=[],
            run_metadata={},
        )
        pack = {
            "query": "dense retrieval",
            "retrieval_mode": "offline",
            "evidence_snippets": [{"source_id": "p1", "snippet": "evidence"}],
        }
        calls: list[tuple[str, int]] = []

        def search_offline(query: str, top_k: int) -> dict:
            calls.append((query, top_k))
            return pack

        result_pack, plan = run_react_topic_rag(
            "Tell me about dense retrieval",
            search_offline=search_offline,
            runtime=runtime,
            model="qwen3:8b",
        )
        self.assertTrue(plan.used_react)
        self.assertEqual(calls, [("dense retrieval", 4)])
        self.assertEqual(result_pack["query"], "dense retrieval")


if __name__ == "__main__":
    unittest.main()
