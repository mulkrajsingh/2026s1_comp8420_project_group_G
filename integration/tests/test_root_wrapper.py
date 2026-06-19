"""Tests for the cross-platform repository-root CLI launcher."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
RPA_PATH = REPO_ROOT / "rpa.py"
SPEC = importlib.util.spec_from_file_location("rpa_launcher", RPA_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load launcher: {RPA_PATH}")
RPA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RPA)


class RootWrapperTests(unittest.TestCase):
    def test_pdf_path_is_resolved_from_callers_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caller = Path(temp_dir)
            paper = caller / "paper with spaces.pdf"
            paper.write_bytes(b"%PDF-1.4\n")

            args = RPA.normalize_args(
                ["peer-review", "paper with spaces.pdf"],
                caller,
            )

            self.assertEqual(args[0], "peer-review")
            self.assertEqual(Path(args[1]), paper.resolve())

    def test_path_options_are_resolved_from_callers_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caller = Path(temp_dir)
            paper = caller / "paper.json"
            corpus = caller / "corpus.jsonl"
            paper.write_text("{}", encoding="utf-8")
            corpus.write_text("{}\n", encoding="utf-8")

            args = RPA.normalize_args(
                [
                    "chat",
                    "What is the contribution?",
                    "--paper-json",
                    "paper.json",
                    "--corpus=corpus.jsonl",
                ],
                caller,
            )

            option_index = args.index("--paper-json")
            self.assertEqual(Path(args[option_index + 1]), paper.resolve())
            self.assertEqual(args[-1], f"--corpus={corpus.resolve()}")

    def test_missing_relative_path_is_preserved_for_cli_error_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caller = Path(temp_dir)
            args = RPA.normalize_args(
                ["analyze-pdf", "missing.pdf"],
                caller,
            )
            self.assertEqual(args[1], "missing.pdf")

    def test_output_paths_are_resolved_before_launcher_changes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caller = Path(temp_dir)
            args = RPA.normalize_args(
                [
                    "recommend-topic",
                    "--out",
                    "results with spaces/recommendations.json",
                    "--report=reports/summary.json",
                ],
                caller,
            )

            output_index = args.index("--out")
            self.assertEqual(
                Path(args[output_index + 1]),
                (caller / "results with spaces" / "recommendations.json").resolve(),
            )
            self.assertEqual(
                args[-1],
                f"--report={(caller / 'reports' / 'summary.json').resolve()}",
            )

    def test_windows_absolute_paths_are_not_rewritten_on_other_platforms(self) -> None:
        windows_path = r"C:\Research Papers\paper.pdf"
        args = RPA.normalize_args(
            ["analyze-pdf", windows_path],
            REPO_ROOT,
        )
        self.assertEqual(args[1], windows_path)

    def test_main_uses_current_interpreter_and_integration_working_directory(self) -> None:
        completed = SimpleNamespace(returncode=7)
        with patch.object(RPA.subprocess, "run", return_value=completed) as run:
            code = RPA.main(["integration-status"], caller_cwd=REPO_ROOT)

        self.assertEqual(code, 7)
        run.assert_called_once_with(
            [sys.executable, "-m", "app.cli", "integration-status"],
            cwd=RPA.INTEGRATION_ROOT,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
