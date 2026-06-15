"""Tests for same-model LLM self-verification."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.cli import _generate_and_verify  # noqa: E402
from app.runtime import GenerationConfig, GenerationResult  # noqa: E402
from app.verification import apply_verification, verify_with_llm  # noqa: E402


class VerificationTests(unittest.TestCase):
    def test_verify_with_llm_uses_revised_answer(self) -> None:
        runtime = Mock()
        runtime.generate.return_value = GenerationResult(
            text=(
                '{"supported": false, "unsupported_claims": ["extra claim"], '
                '"revised_answer": "Grounded answer [2102.00001]"}'
            ),
            backend="ollama",
            model="qwen3:8b",
            prompt_id="llm_self_verification",
            task="self_verification",
            strategy="zero_shot",
            latency_seconds=0.5,
            error=None,
            evidence_ids_used=[],
            run_metadata={},
        )
        pack = {
            "query": "topic",
            "evidence_snippets": [
                {"source_id": "2102.00001", "title": "Paper", "snippet": "evidence"}
            ],
        }
        audit = verify_with_llm(
            "Draft with unsupported claim",
            task="topic_search_synthesis",
            model="qwen3:8b",
            runtime=runtime,
            pack=pack,
        )
        self.assertTrue(audit["verification_used"])
        self.assertTrue(audit["verification_revised"])
        self.assertEqual(audit["revised_answer"], "Grounded answer [2102.00001]")

    def test_apply_verification_keeps_draft_when_disabled(self) -> None:
        result = GenerationResult(
            text="Original draft",
            backend="ollama",
            model="qwen3:8b",
            prompt_id="P02",
            task="topic_search_synthesis",
            strategy="zero_shot",
            latency_seconds=0.1,
            error=None,
            evidence_ids_used=[],
            run_metadata={},
        )
        runtime = Mock()
        with patch.dict(os.environ, {"COMP8420_LLM_VERIFY": "0"}):
            updated, audit = apply_verification(
                result,
                runtime=runtime,
                model="qwen3:8b",
            )
        self.assertEqual(updated.text, "Original draft")
        self.assertFalse(audit["verification_used"])
        runtime.generate.assert_not_called()

    def test_generate_and_verify_can_skip_second_runtime_call(self) -> None:
        result = GenerationResult(
            text='{"summary":"Answer","key_findings":[],"research_gaps":[]}',
            backend="ollama",
            model="qwen3:8b",
            prompt_id="P02",
            task="topic_search_synthesis",
            strategy="zero_shot",
            latency_seconds=0.1,
            error=None,
            evidence_ids_used=[],
            run_metadata={},
        )
        runtime = Mock()
        runtime.generate.return_value = result

        updated, audit = _generate_and_verify(
            runtime,
            {"task": "topic_search_synthesis"},
            "qwen3:8b",
            "zero_shot",
            Mock(),
            verify=False,
        )

        self.assertIs(updated, result)
        self.assertEqual(runtime.generate.call_count, 1)
        self.assertFalse(audit["verification_used"])
        self.assertEqual(audit["verification_reason"], "disabled_by_cli")

    def test_fast_verifier_uses_one_inline_verified_generation(self) -> None:
        draft = GenerationResult(
            text=(
                '{"summary":"Grounded answer [2102.00001]",'
                '"key_findings":[],"research_gaps":[]}'
            ),
            backend="ollama",
            model="qwen3:8b",
            prompt_id="P02",
            task="topic_search_synthesis",
            strategy="zero_shot",
            latency_seconds=0.1,
            error=None,
            evidence_ids_used=["2102.00001"],
            run_metadata={},
        )
        runtime = Mock()
        runtime.generate.return_value = draft

        updated, audit = _generate_and_verify(
            runtime,
            {
                "task": "topic_search_synthesis",
                "zero_shot_prompt": "Answer from the evidence.",
                "few_shot_prompt": "Answer from the examples and evidence.",
            },
            "qwen3:8b",
            "zero_shot",
            GenerationConfig(),
            pack={
                "query": "topic",
                "evidence_snippets": [
                    {
                        "source_id": "2102.00001",
                        "title": "Paper",
                        "snippet": "Evidence",
                    }
                ],
            },
            fast_verify=True,
        )

        self.assertTrue(audit["verification_used"])
        self.assertTrue(audit["supported"])
        self.assertEqual(audit["verification_mode"], "inline_same_model")
        self.assertEqual(runtime.generate.call_count, 1)
        generation_record = runtime.generate.call_args.args[0]
        self.assertIn(
            "verify every factual claim",
            generation_record["zero_shot_prompt"],
        )
        self.assertEqual(
            updated.run_metadata["llm_verification"]["verification_mode"],
            "inline_same_model",
        )

    def test_inline_verifier_does_not_claim_failed_generation_was_verified(self) -> None:
        failed = GenerationResult(
            text="",
            backend="ollama",
            model="qwen3:8b",
            prompt_id="P02",
            task="topic_search_synthesis",
            strategy="zero_shot",
            latency_seconds=0.1,
            error="connection failed",
            evidence_ids_used=[],
            run_metadata={},
        )
        runtime = Mock()
        runtime.generate.return_value = failed

        updated, audit = _generate_and_verify(
            runtime,
            {
                "task": "topic_search_synthesis",
                "zero_shot_prompt": "Answer from the evidence.",
                "few_shot_prompt": "Answer from the examples and evidence.",
            },
            "qwen3:8b",
            "zero_shot",
            GenerationConfig(),
            pack={"query": "topic", "evidence_snippets": []},
            fast_verify=True,
        )

        self.assertIs(updated, failed)
        self.assertFalse(audit["verification_used"])
        self.assertEqual(audit["error"], "connection failed")


if __name__ == "__main__":
    unittest.main()
