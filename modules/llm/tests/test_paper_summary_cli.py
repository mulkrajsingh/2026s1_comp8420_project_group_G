"""CLI tests for paper-only summarization and RAG synthesis compatibility."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.cli import main
from app.prompt_library import fixed_prompt_records, paper_review_prompt_record
from app.query_understanding import QueryAnalysis
from app.runtime import (
    CHIT_CHAT_CONTEXT_TOKENS,
    CHIT_CHAT_MAX_NEW_TOKENS,
    LONG_CONTEXT_MAX_NEW_TOKENS,
    LONG_CONTEXT_TOKENS,
    TECHNICAL_CONTEXT_TOKENS,
    GenerationResult,
    _policy_metadata,
    generation_policy_for_record,
)

PARSED_ARTIFACT = REPO_ROOT / "tests" / "papers" / "artifacts" / "drq_v2_parsed.json"
RAG_ARTIFACT = REPO_ROOT / "tests" / "papers" / "artifacts" / "drq_v2_rag_pack.json"


def _parsed_paper_dict() -> dict:
    if PARSED_ARTIFACT.is_file():
        return json.loads(PARSED_ARTIFACT.read_text(encoding="utf-8"))
    return {
        "metadata": {
            "paper_id": "uploaded_test",
            "title": "Image Augmentation Is All You Need",
            "abstract": "Deep reinforcement learning from pixels.",
            "authors": ["Kostrikov", "Nair", "Zhou"],
            "categories": ["cs.LG"],
            "published_date": "2021-07-20",
            "venue": None,
            "doi": None,
            "arxiv_id": "2107.09645",
            "url": "https://arxiv.org/abs/2107.09645",
            "source": "uploaded_pdf",
        },
        "sections": {
            "abstract": "Deep reinforcement learning from pixels.",
            "introduction": "Introduction.",
            "method": "Method.",
            "results": "Results.",
            "conclusion": "Conclusion.",
        },
        "references": ["Reference."],
        "keywords": ["reinforcement learning"],
        "entities": {
            "methods": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "institutions": [],
        },
    }


def _stub_analyze_query(
    query: str,
    *,
    style_override: str = "auto",
) -> QueryAnalysis:
    chit_chat = query.strip().lower() in {"hi", "hello", "thanks"}
    style = "concise" if style_override == "auto" else style_override
    style_source = (
        "cosine_similarity"
        if style_override == "auto"
        else "explicit_override"
    )
    return QueryAnalysis(
        intent="chit_chat" if chit_chat else "question",
        emotion="positive" if chit_chat else "neutral",
        topic_expertise="intermediate",
        verbosity="concise" if chit_chat else "normal",
        style=style,
        confidence=0.9,
        style_source=style_source,
        style_scores={style: 0.9},
        field_confidences={
            "intent": 0.9,
            "emotion": 0.9,
            "topic_expertise": 0.9,
            "verbosity": 0.9,
            "style": 0.9,
        },
        field_sources={
            "intent": "cosine_similarity",
            "emotion": "cosine_similarity",
            "topic_expertise": "cosine_similarity",
            "verbosity": "cosine_similarity",
            "style": style_source,
        },
        embedding_model="test/minilm",
    )


def _mock_ollama_generate(prompt_record, model, strategy, config):
    task = prompt_record["task"]
    policy = generation_policy_for_record(prompt_record, config)
    run_metadata = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        **_policy_metadata(policy),
    }
    if task == "uploaded_paper_summary":
        text = (
            "## Scope\nTest scope.\n\n"
            "## Core Contribution\n- Main pixel-based RL contribution.\n\n"
            "## Method\nMethod section.\n\n"
            "## Results\n- Strong benchmark results.\n\n"
            "## Limitations\n- Evaluation scope is limited."
        )
    elif task == "peer_review_critique":
        text = (
            "## Strengths\n- Clear structure.\n\n"
            "## Weaknesses\n- Limited evaluation.\n\n"
            "## Missing Evidence\n- Human review needed.\n\n"
            "## Suggested Improvements\n- Add baselines.\n\n"
            "## Evidence Basis\n- abstract, method."
        )
    elif task == "direct_text_chat":
        text = "Hello! Happy to help."
    elif task == "paper_question_answer":
        text = "The paper compares against DrQ and CURL baselines."
    elif task == "topic_search_synthesis":
        text = json.dumps(
            {
                "summary": "Topic synthesis summary [S1].",
                "key_findings": ["Finding [S1]", "Finding [S2]"],
                "research_gaps": ["Gap [S3]"],
            }
        )
    else:
        text = f"Mock output for {task}."
    return GenerationResult(
        text=text,
        backend="ollama",
        model=model,
        prompt_id=prompt_record["prompt_id"],
        task=task,
        strategy=strategy,
        latency_seconds=0.01,
        error=None,
        evidence_ids_used=[],
        run_metadata=run_metadata,
    )


class PaperSummaryCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> None:
        with (
            patch.object(sys, "argv", ["python -m app.cli", *args]),
            patch("app.cli.analyze_query", side_effect=_stub_analyze_query),
            patch(
                "app.runtime.OllamaRuntime.generate",
                side_effect=_mock_ollama_generate,
            ),
        ):
            main()

    def test_successful_paper_only_summary_without_rag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_path = root / "paper.json"
            out_path = root / "summary.md"
            paper_path.write_text(json.dumps(_parsed_paper_dict()), encoding="utf-8")

            with patch(
                "app.cli.topic_synthesis_prompt_record",
                side_effect=AssertionError("summarize must not load sample RAG data"),
            ):
                self.run_cli(
                    "summarize",
                    "--paper",
                    str(paper_path),
                    "--out",
                    str(out_path),
                    "--style",
                    "concise",
                )

            metadata_path = root / "summary_generation.json"
            result_path = root / "summary_result.json"
            markdown = out_path.read_text(encoding="utf-8")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            structured = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["backend"], "ollama")
            self.assertEqual(metadata["input_contract"], "parsed_paper_only")
            self.assertNotIn("# Paper Summary", structured["summary"])
            self.assertTrue(structured["key_findings"])
            self.assertTrue(structured["research_gaps"])
            self.assertFalse(metadata["retrieval_used"])
            self.assertFalse(metadata["external_evidence_used"])
            self.assertEqual(metadata["evidence_ids_used"], [])
            self.assertEqual(metadata["requested_style"], "concise")
            self.assertEqual(metadata["resolved_style"], "concise")
            self.assertTrue(metadata["run_metadata"]["thinking_enabled"])
            self.assertEqual(
                metadata["run_metadata"]["thinking_policy_reason"],
                "long_context_thinking:uploaded_paper_summary",
            )
            self.assertEqual(
                metadata["run_metadata"]["context_window"],
                LONG_CONTEXT_TOKENS,
            )
            self.assertEqual(
                metadata["run_metadata"]["max_new_tokens"],
                min(2048, LONG_CONTEXT_MAX_NEW_TOKENS),
            )

    def test_peer_review_uses_paper_only_few_shot_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_path = root / "paper.json"
            out_path = root / "review.md"
            paper = _parsed_paper_dict()
            paper.setdefault("analysis", {})["structural_checks"] = [
                {"message": "No explicit limitations section was detected."}
            ]
            paper_path.write_text(json.dumps(paper), encoding="utf-8")

            self.run_cli(
                "review",
                "--paper",
                str(paper_path),
                "--out",
                str(out_path),
                "--style",
                "reviewer",
            )

            structured = json.loads(
                (root / "review_result.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (root / "review_generation.json").read_text(encoding="utf-8")
            )
            review = structured["peer_review"]
            self.assertIn("## Strengths", review)
            self.assertIn("## Weaknesses", review)
            self.assertIn("## Missing Evidence", review)
            self.assertEqual(metadata["task"], "peer_review_critique")
            self.assertEqual(metadata["strategy"], "few_shot")
            self.assertEqual(metadata["prompt_strategy"], "few_shot")
            self.assertFalse(metadata["retrieval_used"])
            self.assertTrue(metadata["run_metadata"]["thinking_enabled"])
            self.assertEqual(
                metadata["run_metadata"]["thinking_policy_reason"],
                "long_context_thinking:peer_review_critique",
            )

    def test_peer_review_prompt_excludes_rag_and_uses_example(self) -> None:
        record = paper_review_prompt_record(_parsed_paper_dict(), "reviewer")

        self.assertEqual(record["task"], "peer_review_critique")
        self.assertIsNone(record["input"]["rag_evidence_pack"])
        self.assertIn("Few-shot example", record["few_shot_prompt"])
        self.assertNotEqual(record["zero_shot_prompt"], record["few_shot_prompt"])
        self.assertTrue(
            record["expected_output_contract"]["must_ground_critique_in_paper"]
        )
        signals = record["input"]["review_presence_signals"]
        self.assertTrue(signals["episode_return_or_reward_metrics"])
        self.assertTrue(signals["hyperparameter_details"])
        self.assertTrue(signals["ablation_study"])
        self.assertTrue(signals["model_free_baseline_comparison"])
        self.assertTrue(signals["model_based_baseline_comparison"])
        inventory = record["input"]["review_evidence_inventory"]
        self.assertIn("episode return", inventory["reported_metrics"])
        self.assertIn("SAC", inventory["named_baselines_or_backbones"])
        self.assertIn("CURL", inventory["named_baselines_or_backbones"])
        self.assertIn("DrQ", inventory["named_baselines_or_backbones"])
        self.assertIn("ablation study", inventory["experiment_details"])
        self.assertIn(
            "Known-present evidence inventory",
            record["few_shot_prompt"],
        )
        self.assertNotIn(
            "omits evaluation metrics",
            record["few_shot_prompt"],
        )

    def test_direct_chat_generates_without_rag_or_parsed_paper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_path = root / "chat.md"

            self.run_cli(
                "chat",
                "--query",
                "hi",
                "--out",
                str(out_path),
                "--style",
                "auto",
            )

            metadata = json.loads(
                (root / "chat_generation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(out_path.read_text(encoding="utf-8").strip())
            self.assertEqual(metadata["input_contract"], "text_only")
            self.assertFalse(metadata["retrieval_used"])
            self.assertFalse(metadata["external_evidence_used"])
            self.assertEqual(metadata["query_analysis"]["intent"], "chit_chat")
            self.assertFalse(metadata["run_metadata"]["thinking_enabled"])
            self.assertEqual(
                metadata["run_metadata"]["context_window"],
                CHIT_CHAT_CONTEXT_TOKENS,
            )
            self.assertEqual(
                metadata["run_metadata"]["max_new_tokens"],
                CHIT_CHAT_MAX_NEW_TOKENS,
            )

    def test_direct_chat_reuses_precomputed_query_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_path = root / "chat.md"
            analysis_path = root / "query_analysis.json"
            analysis_path.write_text(
                json.dumps(_stub_analyze_query("hi").as_dict()),
                encoding="utf-8",
            )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "python -m app.cli",
                        "chat",
                        "--query",
                        "hi",
                        "--style",
                        "concise",
                        "--query-analysis",
                        str(analysis_path),
                        "--out",
                        str(out_path),
                    ],
                ),
                patch(
                    "app.cli.analyze_query",
                    side_effect=AssertionError("query must not be classified twice"),
                ),
                patch(
                    "app.runtime.OllamaRuntime.generate",
                    side_effect=_mock_ollama_generate,
                ),
            ):
                main()

            metadata = json.loads(
                (root / "chat_generation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["query_analysis"]["intent"], "chit_chat")

            paper_path = root / "paper.json"
            question_out = root / "question.md"
            question_analysis = _stub_analyze_query("question").as_dict()
            question_analysis["intent"] = "question"
            paper_path.write_text(
                json.dumps(_parsed_paper_dict()),
                encoding="utf-8",
            )
            analysis_path.write_text(
                json.dumps(question_analysis),
                encoding="utf-8",
            )
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "python -m app.cli",
                        "summarize",
                        "--paper",
                        str(paper_path),
                        "--query",
                        "Which baseline was used?",
                        "--style",
                        "concise",
                        "--query-analysis",
                        str(analysis_path),
                        "--out",
                        str(question_out),
                    ],
                ),
                patch(
                    "app.cli.analyze_query",
                    side_effect=AssertionError("query must not be classified twice"),
                ),
                patch(
                    "app.runtime.OllamaRuntime.generate",
                    side_effect=_mock_ollama_generate,
                ),
            ):
                main()

            question_metadata = json.loads(
                (root / "question_generation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(question_metadata["task"], "paper_question_answer")

    @unittest.skipUnless(
        PARSED_ARTIFACT.is_file() and RAG_ARTIFACT.is_file(),
        "run tests/harness/bootstrap_artifacts.py first",
    )
    def test_fixed_uploaded_paper_prompt_is_paper_only(self) -> None:
        record = fixed_prompt_records()[0]

        self.assertEqual(record["task"], "uploaded_paper_summary")
        self.assertIsNone(record["input"]["rag_evidence_pack"])
        self.assertFalse(record["expected_output_contract"]["must_include_source_ids"])

    def test_missing_paper_file_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.json"
            out_path = Path(temp_dir) / "summary.md"

            with self.assertRaisesRegex(SystemExit, "ParsedPaper file not found"):
                self.run_cli(
                    "summarize",
                    "--paper",
                    str(missing_path),
                    "--out",
                    str(out_path),
                )

    def test_invalid_paper_schema_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_path = root / "invalid.json"
            paper_path.write_text('{"metadata": {}}', encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "Schema validation failed"):
                self.run_cli(
                    "summarize",
                    "--paper",
                    str(paper_path),
                    "--out",
                    str(root / "summary.md"),
                )

    def test_missing_rag_evidence_has_no_sample_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_path = root / "synthesis.md"

            with self.assertRaisesRegex(
                SystemExit, "RagEvidencePack file not found"
            ):
                self.run_cli(
                    "synthesize",
                    "--evidence",
                    str(root / "missing-evidence.json"),
                    "--out",
                    str(out_path),
                )

            self.assertFalse(out_path.exists())

    def test_backend_error_is_written_and_reported(self) -> None:
        class ErrorRuntime:
            def generate(self, prompt_record, model, strategy, config):
                return GenerationResult(
                    text="",
                    backend="ollama",
                    model=model,
                    prompt_id=prompt_record["prompt_id"],
                    task=prompt_record["task"],
                    strategy=strategy,
                    latency_seconds=0.1,
                    error="connection refused",
                    evidence_ids_used=[],
                    run_metadata={},
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_path = root / "paper.json"
            out_path = root / "summary.md"
            paper_path.write_text(json.dumps(_parsed_paper_dict()), encoding="utf-8")

            with patch("app.cli.build_runtime", return_value=ErrorRuntime()):
                with self.assertRaisesRegex(
                    SystemExit,
                    "Generation backend reported an error: connection refused",
                ):
                    self.run_cli(
                        "summarize",
                        "--paper",
                        str(paper_path),
                        "--out",
                        str(out_path),
                    )

            self.assertIn(
                "Generation failed: connection refused",
                out_path.read_text(encoding="utf-8"),
            )
            metadata = json.loads(
                (root / "summary_generation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["error"], "connection refused")
            self.assertFalse(metadata["retrieval_used"])

    @unittest.skipUnless(
        RAG_ARTIFACT.is_file(),
        "run tests/harness/bootstrap_artifacts.py first",
    )
    def test_existing_rag_synthesis_remains_functional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_path = root / "evidence.json"
            out_path = root / "rag-summary.md"
            json_out = root / "analysis.json"
            evidence_path.write_text(
                RAG_ARTIFACT.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            self.run_cli(
                "synthesize",
                "--evidence",
                str(evidence_path),
                "--out",
                str(out_path),
                "--json-out",
                str(json_out),
            )

            markdown = out_path.read_text(encoding="utf-8")
            analysis = json.loads(json_out.read_text(encoding="utf-8"))
            generation = json.loads(
                (root / "llm_generation.json").read_text(encoding="utf-8")
            )
            self.assertIn("Retrieval mode", markdown)
            self.assertEqual(analysis["input_type"], "topic_text")
            self.assertTrue(analysis["recommendations"])
            self.assertEqual(generation["requested_style"], "auto")
            self.assertEqual(generation["resolved_style"], "concise")
            self.assertEqual(
                generation["query_analysis"]["style_source"],
                "cosine_similarity",
            )
            self.assertFalse(generation["run_metadata"]["thinking_enabled"])
            self.assertEqual(
                generation["run_metadata"]["thinking_policy_reason"],
                "structured_direct:topic_search_synthesis",
            )
            self.assertEqual(generation["run_metadata"]["max_new_tokens"], 768)
            self.assertEqual(
                generation["run_metadata"]["context_window"],
                TECHNICAL_CONTEXT_TOKENS,
            )


if __name__ == "__main__":
    unittest.main()
