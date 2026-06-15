"""Tests for delegated member commands."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]


class MemberCommandDelegationTests(unittest.TestCase):
    def _help(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "app.cli", *args, "--help"],
            cwd=INTEGRATION_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_retrieval_evaluate_help_delegates(self) -> None:
        proc = self._help("evaluate-retrieval")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("evaluate-retrieval", proc.stdout)

    def test_llm_compare_models_help_delegates(self) -> None:
        proc = self._help("compare-models")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("compare-models", proc.stdout)

    def test_pdf_nlp_parse_help_delegates(self) -> None:
        proc = self._help("parse-pdf")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("parse-pdf", proc.stdout)

    def test_dataset_classify_help_delegates(self) -> None:
        proc = self._help("classify-domains")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("domain classifier", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
