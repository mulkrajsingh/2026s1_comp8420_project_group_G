"""Tests for the traditional paper-category classifier baseline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from scripts.build_balanced_corpus import TARGET_CATEGORIES  # noqa: E402
from scripts.evaluate_domain_classifier import evaluate_domain_classifier  # noqa: E402


class DomainClassifierTests(unittest.TestCase):
    def test_writes_reproducible_metric_artifacts(self) -> None:
        vocabulary = {
            "cs.AI": "planning reasoning search agents",
            "cs.CL": "language translation tokens parsing",
            "cs.LG": "learning neural optimization training",
            "stat.ML": "bayesian statistics inference probability",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "papers.jsonl"
            rows = []
            for category in TARGET_CATEGORIES:
                for index in range(12):
                    rows.append(
                        {
                            "paper_id": f"{category}-{index}",
                            "title": f"{category} study {index}",
                            "abstract": (
                                f"{vocabulary[category]} methods and results "
                                f"for category example {index}"
                            ),
                            "categories": [category],
                        }
                    )
            corpus.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output_dir = root / "results"
            first = evaluate_domain_classifier(
                corpus,
                output_dir,
                test_size=0.25,
                seed=7,
                max_features=1_000,
            )
            second = evaluate_domain_classifier(
                corpus,
                output_dir,
                test_size=0.25,
                seed=7,
                max_features=1_000,
            )

            self.assertEqual(first["accuracy"], second["accuracy"])
            self.assertEqual(first["macro_f1"], second["macro_f1"])
            self.assertEqual(first["test_records"], 12)
            self.assertGreater(first["accuracy"], first["majority_baseline_accuracy"])
            self.assertEqual(
                set(first["model_comparison"]),
                {"logistic_regression", "linear_svm"},
            )
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "confusion_matrix.csv").is_file())
            self.assertTrue((output_dir / "confusion_matrix.png").is_file())
            self.assertTrue((output_dir / "linear_svm_confusion_matrix.csv").is_file())
            self.assertTrue((output_dir / "linear_svm_confusion_matrix.png").is_file())
            self.assertTrue((output_dir / "model_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
