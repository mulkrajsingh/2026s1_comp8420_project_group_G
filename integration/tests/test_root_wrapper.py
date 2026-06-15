"""Tests for the repository-root CLI wrapper."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RPA = REPO_ROOT / "scripts" / "rpa"


class RootWrapperTests(unittest.TestCase):
    def test_pdf_path_is_resolved_from_callers_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper.pdf"
            paper.write_bytes(b"%PDF-1.4\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            capture = root / "args.txt"
            python = bin_dir / "python"
            python.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$RPA_CAPTURE\"\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "RPA_CAPTURE": str(capture),
            }

            subprocess.run(
                [str(RPA), "peer-review", "paper.pdf"],
                cwd=root,
                env=env,
                check=True,
            )

            args = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(args[:3], ["-m", "app.cli", "peer-review"])
            self.assertEqual(Path(args[3]).resolve(), paper.resolve())

    def test_paper_json_option_is_resolved_from_callers_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper.json"
            paper.write_text("{}", encoding="utf-8")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            capture = root / "args.txt"
            python = bin_dir / "python"
            python.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$RPA_CAPTURE\"\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "RPA_CAPTURE": str(capture),
            }

            subprocess.run(
                [
                    str(RPA),
                    "chat",
                    "What is the contribution?",
                    "--paper-json",
                    "paper.json",
                ],
                cwd=root,
                env=env,
                check=True,
            )

            args = capture.read_text(encoding="utf-8").splitlines()
            option_index = args.index("--paper-json")
            self.assertEqual(Path(args[option_index + 1]).resolve(), paper.resolve())


if __name__ == "__main__":
    unittest.main()
