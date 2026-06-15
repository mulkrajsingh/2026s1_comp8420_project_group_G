"""Tests for production and synthetic evidence source IDs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.faithfulness import check_generation, output_source_ids  # noqa: E402
from app.runtime import source_ids_used  # noqa: E402


class FaithfulnessSourceIdTests(unittest.TestCase):
    def test_extracts_bare_arxiv_source_id(self) -> None:
        self.assertEqual(
            output_source_ids("Recommended source: 2102.00002"),
            {"2102.00002"},
        )

    def test_extracts_arxiv_and_synthetic_ids(self) -> None:
        text = (
            "Grounded claim [2604.26126]. Combined evidence [S1, S2]. "
            "Parsed paper reference [2107.09645v1_fd56aa5c6a]. "
            "Normal [paper title](https://example.test) is not a source ID."
        )

        expected = {"2604.26126", "2107.09645v1_fd56aa5c6a", "S1", "S2"}
        self.assertEqual(output_source_ids(text), expected)
        self.assertEqual(set(source_ids_used(text)), expected)

    def test_faithfulness_accepts_production_arxiv_id(self) -> None:
        pack = {
            "evidence_snippets": [
                {"source_id": "2604.26126"},
                {"source_id": "2605.01520"},
            ],
            "candidates": [],
        }

        result = check_generation(
            "The controller reduces communication frequency [2604.26126].",
            pack,
        )

        self.assertTrue(result["passes_basic_faithfulness"])
        self.assertEqual(result["used_source_ids"], ["2604.26126"])
        self.assertEqual(result["unsupported_source_ids"], [])


if __name__ == "__main__":
    unittest.main()
