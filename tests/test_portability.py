"""Repository-level checks for platform-neutral entrypoints and instructions."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}
CANONICAL_TEXT_FILES = [
    REPO_ROOT / "readme.md",
    *sorted((REPO_ROOT / "docs").glob("*.md")),
    REPO_ROOT / "integration" / "README.md",
    REPO_ROOT / "integration" / "frontend" / "README.md",
    REPO_ROOT / "integration" / "data" / "sessions" / "README.md",
    REPO_ROOT / "modules" / "dataset" / "README.md",
    REPO_ROOT / "modules" / "pdf_nlp" / "README.md",
    REPO_ROOT / "modules" / "retrieval" / "README.md",
    REPO_ROOT / "modules" / "llm" / "README.md",
    REPO_ROOT / "modules" / "llm" / "lora_dataset" / "README.md",
    REPO_ROOT / "tests" / "README.md",
    REPO_ROOT / "tests" / "TEST_CASES.md",
    REPO_ROOT / "setup_assets.py",
    REPO_ROOT / "integration" / "app" / "api.py",
    REPO_ROOT / "integration" / "app" / "docs_gen.py",
    REPO_ROOT / "integration" / "app" / "frontend.py",
]
LEGACY_REFERENCES = (
    "scripts/rpa",
    "run_system_tests.sh",
    "smoke_test_all.sh",
    "create_dataset.sh",
)


def _source_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        yield path


class PortabilityTests(unittest.TestCase):
    def test_repository_has_no_shell_entrypoints(self) -> None:
        shell_files = [
            path.relative_to(REPO_ROOT)
            for path in _source_files()
            if path.suffix.lower() == ".sh"
        ]
        self.assertEqual(shell_files, [])

    def test_canonical_instructions_do_not_reference_removed_entrypoints(self) -> None:
        findings: list[str] = []
        for path in CANONICAL_TEXT_FILES:
            text = path.read_text(encoding="utf-8")
            for legacy in LEGACY_REFERENCES:
                if legacy in text:
                    findings.append(f"{path.relative_to(REPO_ROOT)}: {legacy}")
        self.assertEqual(findings, [])

    def test_python_entrypoints_compile(self) -> None:
        for relative in (
            "rpa.py",
            "scripts/smoke_test_all.py",
            "tests/run_system_tests.py",
        ):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            compile(source, relative, "exec")


if __name__ == "__main__":
    unittest.main()
