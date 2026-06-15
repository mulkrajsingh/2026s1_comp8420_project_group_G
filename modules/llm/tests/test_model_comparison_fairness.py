"""Regression tests for fair base-versus-adapter evaluation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.evaluation import _score_output, compare_models, ensure_fixed_prompts  # noqa: E402
from app.io_utils import read_jsonl  # noqa: E402
from app.runtime import GenerationResult, prompt_text_for_record  # noqa: E402


class ModelComparisonFairnessTests(unittest.TestCase):
    def test_paper_only_task_is_not_penalised_for_missing_rag_ids(self) -> None:
        score = _score_output(
            "# Summary\nGrounded paper summary.",
            "uploaded_paper_summary",
            "few_shot",
            None,
        )

        self.assertFalse(score["evidence_check_applicable"])
        self.assertEqual(score["evidence_faithfulness"], 1.0)

    def test_supplied_doi_url_is_citation_safe(self) -> None:
        pack = {
            "evidence_snippets": [{"source_id": "2102.00002"}],
            "candidates": [
                {
                    "paper": {
                        "doi": "10.1234/xyz",
                        "url": "https://arxiv.org/abs/2102.00002",
                    },
                    "apa_citation": (
                        "Turing, A. (2021). Paper. "
                        "https://doi.org/10.1234/xyz"
                    ),
                }
            ],
        }
        score = _score_output(
            "Recommended [2102.00002]: https://doi.org/10.1234/xyz",
            "citation_recommendation",
            "few_shot",
            pack,
        )

        self.assertEqual(score["citation_safety"], 1.0)
        self.assertEqual(score["evidence_faithfulness"], 1.0)

    def test_fixed_prompts_match_production_input_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_set = Path(temp_dir) / "fixed_prompts.jsonl"
            ensure_fixed_prompts(test_set)
            records = {
                record["prompt_id"]: record
                for record in read_jsonl(test_set)
            }

            self.assertIsNone(records["P04"]["input"]["parsed_paper"])
            self.assertIsNotNone(records["P04"]["input"]["rag_evidence_pack"])
            self.assertIn(
                "Do not recommend weakly related or unrelated candidates",
                records["P04"]["few_shot_prompt"],
            )
            self.assertIn(
                "Broad field overlap alone is insufficient",
                records["P04"]["few_shot_prompt"],
            )
            self.assertIsNotNone(records["P05"]["input"]["parsed_paper"])
            self.assertIsNone(records["P05"]["input"]["rag_evidence_pack"])
            self.assertEqual(records["P05"]["task"], "peer_review_critique")

            citation_prompt = prompt_text_for_record(records["P04"], "few_shot")
            self.assertNotIn('"prompt_templates"', citation_prompt)
            self.assertNotIn('"reason"', citation_prompt)
            self.assertNotIn('"score"', citation_prompt)
            self.assertIn('"eligible_candidates": []', citation_prompt)
            self.assertIn('"nearby_candidates"', citation_prompt)
            self.assertIn('"source_id": "2102.00002"', citation_prompt)

    def test_models_use_the_same_prompt_strategy(self) -> None:
        def _mock_generate(prompt_record, model, strategy, config):
            return GenerationResult(
                text=f"Mock output for {prompt_record['prompt_id']}.",
                backend="ollama",
                model=model,
                prompt_id=prompt_record["prompt_id"],
                task=prompt_record["task"],
                strategy=strategy,
                latency_seconds=0.01,
                error=None,
                evidence_ids_used=[],
                run_metadata={},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            test_set = root / "fixed_prompts.jsonl"
            ensure_fixed_prompts(test_set)
            output_dir = root / "comparison"
            with patch(
                "app.runtime.OllamaRuntime.generate",
                side_effect=_mock_generate,
            ):
                compare_models(
                    test_set,
                    output_dir,
                    model_specs=[
                    {
                        "name": "base",
                        "model": "base",
                        "adapter": "none",
                        "quantization": "test",
                        "is_adapter": "false",
                    },
                    {
                        "name": "lora",
                        "model": "lora",
                        "adapter": "adapter",
                        "quantization": "test",
                        "is_adapter": "true",
                    },
                ],
                )
            with (output_dir / "final_model_comparison.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 12)
            self.assertEqual({row["prompt_strategy"] for row in rows}, {"few_shot"})
            for prompt_id in {row["prompt_id"] for row in rows}:
                prompt_rows = [row for row in rows if row["prompt_id"] == prompt_id]
                self.assertEqual(
                    {row["prompt_strategy"] for row in prompt_rows},
                    {"few_shot"},
                )
            self.assertFalse(
                (output_dir / "llm_as_judge_scores.csv").exists()
            )
            self.assertFalse(
                (root / "adapter_training_manifest.md").exists()
            )


if __name__ == "__main__":
    unittest.main()
