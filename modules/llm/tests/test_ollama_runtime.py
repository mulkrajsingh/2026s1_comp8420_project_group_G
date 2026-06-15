"""Tests for Ollama generation request metadata."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.prompt_library import (
    direct_chat_prompt_record,
    paper_recommendation_prompt_record,
    paper_review_prompt_record,
    paper_summary_prompt_record,
    topic_synthesis_prompt_record,
)
from app.runtime import (
    CHIT_CHAT_CONTEXT_TOKENS,
    CHIT_CHAT_MAX_NEW_TOKENS,
    LONG_CONTEXT_THINKING_MAX_NEW_TOKENS,
    LONG_CONTEXT_TOKENS,
    TECHNICAL_CONTEXT_TOKENS,
    TECHNICAL_MAX_NEW_TOKENS,
    GenerationConfig,
    OllamaRuntime,
    generation_policy_for_record,
    prompt_text_for_record,
)


class _Response:
    def __init__(self, response_text: str = "Hello") -> None:
        self._response_text = response_text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "response": self._response_text,
                "load_duration": 2_000_000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 3_000_000,
                "eval_count": 2,
                "eval_duration": 4_000_000,
            }
        ).encode("utf-8")


class OllamaGenerationTests(unittest.TestCase):
    def test_citation_guard_rejects_nearby_only_candidate(self) -> None:
        record = {
            "prompt_id": "P04",
            "task": "citation_recommendation",
            "style": "technical",
            "zero_shot_prompt": "Recommend citations.",
            "few_shot_prompt": "Recommend citations.",
            "input": {
                "parsed_paper": None,
                "rag_evidence_pack": {
                    "query": (
                        "Mastering Visual Continuous Control: Improved "
                        "Data-Augmented Reinforcement Learning"
                    ),
                    "candidates": [
                        {
                            "paper": {
                                "paper_id": "2102.00002",
                                "title": "Reinforcement Learning for Robotics",
                                "abstract": (
                                    "An RL method for robotic control is introduced."
                                ),
                                "authors": ["Alan Turing"],
                                "published_date": "2021-02-02",
                                "doi": "10.1234/xyz",
                                "url": "https://arxiv.org/abs/2102.00002",
                            },
                            "apa_citation": (
                                "Turing, A. (2021). Reinforcement Learning for "
                                "Robotics. https://doi.org/10.1234/xyz"
                            ),
                        }
                    ],
                    "evidence_snippets": [
                        {
                            "source_id": "2102.00002",
                            "title": "Reinforcement Learning for Robotics",
                            "snippet": (
                                "An RL method for robotic control is introduced."
                            ),
                        }
                    ],
                },
            },
            "expected_output_contract": {},
        }
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response("Recommended: [2102.00002]"),
        ):
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "few_shot",
                GenerationConfig(),
            )

        self.assertIn("No directly relevant citation was found", result.text)
        self.assertIn("Nearby leads (not recommended", result.text)
        self.assertTrue(
            result.run_metadata["citation_eligibility_guard_applied"]
        )

    def test_research_gap_reasoning_has_bounded_budget(self) -> None:
        policy = generation_policy_for_record(
            {"task": "research_gap_identification"},
            GenerationConfig(),
        )

        self.assertTrue(policy.thinking_enabled)
        self.assertEqual(policy.max_new_tokens, 4096)
        self.assertEqual(
            policy.reason,
            "bounded_reasoning:research_gap_identification",
        )

    def test_empty_think_wrapper_is_removed(self) -> None:
        record = direct_chat_prompt_record(
            "hello",
            "concise",
            query_analysis={"intent": "chit_chat"},
        )
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response("<think>\n\n</think>\n\nHello"),
        ):
            result = OllamaRuntime().generate(
                record,
                "qwen3-research-lora:latest",
                "zero_shot",
                GenerationConfig(),
            )

        self.assertEqual(result.text, "Hello")

    def test_web_keep_alive_and_runtime_counters_are_recorded(self) -> None:
        record = direct_chat_prompt_record(
            "hi",
            "concise",
            query_analysis={"intent": "chit_chat"},
        )
        with (
            patch.dict(
                os.environ,
                {"COMP8420_OLLAMA_KEEP_ALIVE": "-1"},
                clear=False,
            ),
            patch(
                "app.ollama_transport.urllib.request.urlopen",
                return_value=_Response(),
            ) as urlopen,
        ):
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(),
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["keep_alive"], -1)
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], CHIT_CHAT_CONTEXT_TOKENS)
        self.assertEqual(
            payload["options"]["num_predict"],
            CHIT_CHAT_MAX_NEW_TOKENS,
        )
        self.assertFalse(result.run_metadata["thinking_enabled"])
        self.assertEqual(result.run_metadata["ollama_load_duration_ms"], 2.0)
        self.assertEqual(result.run_metadata["ollama_prompt_eval_count"], 10)
        self.assertEqual(result.run_metadata["ollama_eval_duration_ms"], 4.0)

    def test_technical_request_enables_thinking_and_maximum_budget(self) -> None:
        record = direct_chat_prompt_record(
            "Explain transformer attention",
            "technical",
            query_analysis={"intent": "request_explanation"},
        )
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response(),
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(),
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], TECHNICAL_CONTEXT_TOKENS)
        self.assertEqual(
            payload["options"]["num_predict"],
            TECHNICAL_MAX_NEW_TOKENS,
        )
        self.assertTrue(result.run_metadata["thinking_enabled"])
        self.assertEqual(
            result.run_metadata["thinking_policy_reason"],
            "technical_task:direct_text_chat",
        )

    def test_paper_summary_enables_thinking_with_higher_cap(self) -> None:
        parsed = {
            "metadata": {"title": "Test", "abstract": "An abstract."},
            "sections": {"abstract": "An abstract.", "method": "A method."},
        }
        record = paper_summary_prompt_record(parsed, "technical")
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response(),
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(),
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], LONG_CONTEXT_TOKENS)
        self.assertEqual(
            payload["options"]["num_predict"],
            min(GenerationConfig().max_new_tokens, LONG_CONTEXT_THINKING_MAX_NEW_TOKENS),
        )
        self.assertEqual(
            result.run_metadata["thinking_policy_reason"],
            "long_context_thinking:uploaded_paper_summary",
        )

    def test_paper_question_enables_thinking_with_higher_cap(self) -> None:
        parsed = {
            "metadata": {"title": "Test", "abstract": "An abstract."},
            "sections": {"results": "Compared against Baseline A."},
        }
        record = paper_summary_prompt_record(
            parsed,
            "concise",
            user_query="Which baseline is compared?",
            query_analysis={"intent": "question", "style": "concise"},
        )
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response(),
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(max_new_tokens=512),
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(record["task"], "paper_question_answer")
        self.assertTrue(payload["think"])
        self.assertEqual(
            payload["options"]["num_predict"],
            min(512, LONG_CONTEXT_THINKING_MAX_NEW_TOKENS),
        )
        self.assertEqual(
            result.run_metadata["thinking_policy_reason"],
            "long_context_thinking:paper_question_answer",
        )

    def test_peer_review_enables_thinking_with_higher_cap(self) -> None:
        parsed = {
            "metadata": {"title": "Test", "abstract": "An abstract."},
            "sections": {
                "abstract": "An abstract.",
                "method": "METHOD_HEAD " + ("x" * 4000) + " METHOD_TAIL_ABLATION",
            },
        }
        record = paper_review_prompt_record(parsed, "reviewer")
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response(),
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "few_shot",
                GenerationConfig(),
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], LONG_CONTEXT_TOKENS)
        self.assertEqual(
            payload["options"]["num_predict"],
            min(GenerationConfig().max_new_tokens, LONG_CONTEXT_THINKING_MAX_NEW_TOKENS),
        )
        self.assertIn('"task": "peer_review_critique"', payload["prompt"])
        self.assertIn("Few-shot example", payload["prompt"])
        self.assertIn("METHOD_HEAD", payload["prompt"])
        self.assertIn("METHOD_TAIL_ABLATION", payload["prompt"])
        self.assertIn('"ablation_study": true', payload["prompt"])
        self.assertEqual(
            result.run_metadata["thinking_policy_reason"],
            "long_context_thinking:peer_review_critique",
        )

    def test_paper_recommendation_uses_direct_structured_generation(self) -> None:
        pack = {
            "query": "retrieval augmented generation",
            "retrieval_mode": "offline",
            "candidates": [],
            "evidence_snippets": [],
        }
        record = paper_recommendation_prompt_record(pack, "concise")
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            return_value=_Response('{"paper_summaries": []}'),
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(),
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertFalse(payload["think"])
        self.assertEqual(
            payload["options"]["num_predict"],
            GenerationConfig().max_new_tokens,
        )
        self.assertEqual(
            result.run_metadata["thinking_policy_reason"],
            "structured_direct:paper_recommendation",
        )

    def test_topic_prompt_does_not_duplicate_candidate_abstracts(self) -> None:
        pack = {
            "query": "retrieval augmented generation",
            "retrieval_mode": "offline",
            "candidates": [
                {
                    "paper": {
                        "paper_id": "p1",
                        "title": "Grounded Generation",
                        "abstract": "DUPLICATE CANDIDATE ABSTRACT",
                        "authors": ["Alice"],
                        "published_date": "2024-01-01",
                        "url": "https://arxiv.org/abs/p1",
                    },
                    "score": 0.9,
                    "relation": "same_topic",
                    "evidence": ["p1"],
                    "apa_citation": "Alice (2024). Grounded Generation.",
                }
            ],
            "evidence_snippets": [
                {
                    "source_id": "p1",
                    "title": "Grounded Generation",
                    "snippet": "Evidence snippet retained for generation.",
                    "metadata": {"year": "2024"},
                }
            ],
        }

        prompt = prompt_text_for_record(
            topic_synthesis_prompt_record(pack, "concise"),
            "zero_shot",
        )

        self.assertNotIn("DUPLICATE CANDIDATE ABSTRACT", prompt)
        self.assertIn("Evidence snippet retained for generation.", prompt)
        self.assertIn("Grounded Generation", prompt)

    def test_react_tool_plan_uses_small_structured_budget(self) -> None:
        policy = generation_policy_for_record(
            {"task": "react_tool_plan"},
            GenerationConfig(),
        )

        self.assertFalse(policy.thinking_enabled)
        self.assertEqual(policy.max_new_tokens, 128)

    def test_empty_thinking_response_retries_without_thinking(self) -> None:
        record = direct_chat_prompt_record(
            "Explain transformer attention",
            "technical",
            query_analysis={"intent": "request_explanation"},
        )
        responses = [_Response(""), _Response("Recovered answer")]
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(),
            )

        self.assertEqual(urlopen.call_count, 2)
        first_payload = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        second_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertTrue(first_payload["think"])
        self.assertFalse(second_payload["think"])
        self.assertEqual(result.text, "Recovered answer")
        self.assertIsNone(result.error)
        self.assertTrue(result.run_metadata["thinking_fallback_used"])

    def test_successful_thinking_response_does_not_retry(self) -> None:
        record = direct_chat_prompt_record(
            "Explain transformer attention",
            "technical",
            query_analysis={"intent": "request_explanation"},
        )
        with patch(
            "app.ollama_transport.urllib.request.urlopen",
            side_effect=[_Response("Direct answer")],
        ) as urlopen:
            result = OllamaRuntime().generate(
                record,
                "qwen3:8b",
                "zero_shot",
                GenerationConfig(),
            )

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(result.text, "Direct answer")
        self.assertFalse(result.run_metadata["thinking_fallback_used"])


if __name__ == "__main__":
    unittest.main()
