"""Tests for deterministic balanced corpus generation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from scripts.build_balanced_corpus import (  # noqa: E402
    TARGET_CATEGORIES,
    TIME_BUCKETS,
    build_balanced_corpus,
)


class BalancedCorpusTests(unittest.TestCase):
    def test_build_is_deterministic_and_balanced(self) -> None:
        years = (2016, 2019, 2021, 2024, 2025)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw.jsonl"
            rows = []
            counter = 0
            for category in TARGET_CATEGORIES:
                for year in years:
                    counter += 1
                    rows.append(
                        {
                            "id": f"{year % 100:02d}01.{counter:04d}",
                            "title": f"{category} paper {year}",
                            "abstract": f"Research abstract for {category} in {year}.",
                            "categories": category,
                            "versions": [
                                {
                                    "version": "v1",
                                    "created": f"Mon, 1 Jan {year} 00:00:00 GMT",
                                }
                            ],
                            "authors_parsed": [["Author", "Test", ""]],
                        }
                    )
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            report = build_balanced_corpus(raw, first, total=20, seed=7)
            build_balanced_corpus(raw, second, total=20, seed=7)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report["written"], 20)
            self.assertEqual(report["year_min"], 2016)
            self.assertEqual(report["year_max"], 2025)
            self.assertEqual(len(TIME_BUCKETS), 4)
            output = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {category for row in output for category in row["categories"]},
                set(TARGET_CATEGORIES),
            )
            self.assertTrue(all(row["url"].startswith("https://arxiv.org/abs/") for row in output))


if __name__ == "__main__":
    unittest.main()
