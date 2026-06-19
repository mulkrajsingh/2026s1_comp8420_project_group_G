"""Cross-platform compile, CLI, fixture, and system-test smoke checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _format_command(command: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


class SmokeRunner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.passed = 0
        self.failed = 0

    def command(
        self,
        name: str,
        command: Sequence[str],
        *,
        cwd: Path = REPO_ROOT,
        env: dict[str, str] | None = None,
    ) -> bool:
        print(f"\n=== {name} ===")
        print(f"cwd: {cwd}")
        print(f"command: {_format_command(command)}")
        if self.dry_run:
            self.passed += 1
            print(f"[DRY RUN] {name}")
            return True
        try:
            code = subprocess.run(
                list(command),
                cwd=cwd,
                env=env,
                check=False,
            ).returncode
        except OSError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            self.failed += 1
            return False
        if code == 0:
            self.passed += 1
            print(f"[PASS] {name}")
            return True
        self.failed += 1
        print(f"[FAIL] {name} (exit {code})", file=sys.stderr)
        return False

    def check(self, name: str, check: Callable[[], None]) -> bool:
        print(f"\n=== {name} ===")
        if self.dry_run:
            self.passed += 1
            print(f"[DRY RUN] {name}")
            return True
        try:
            check()
        except Exception as exc:
            self.failed += 1
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return False
        self.passed += 1
        print(f"[PASS] {name}")
        return True

    def summary(self) -> int:
        print("\n========================================")
        print(f"Smoke summary: {self.passed} passed, {self.failed} failed")
        print("========================================")
        return 0 if self.failed == 0 else 1


def _validate_corpus() -> None:
    path = REPO_ROOT / "modules" / "dataset" / "data" / "processed" / "dev_5k.jsonl"
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
            count += 1
    if count != 5000:
        raise AssertionError(f"expected 5000 records, found {count}")
    print(f"Validated {count} records in {path.relative_to(REPO_ROOT)}")


def _validate_retrieval_output() -> None:
    path = (
        REPO_ROOT
        / "modules"
        / "retrieval"
        / "outputs"
        / "smoke_recommendations.json"
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"missing or empty output: {path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-ollama", action="store_true")
    group.add_argument("--require-ollama", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = SmokeRunner(dry_run=args.dry_run)
    python = sys.executable

    steps: list[tuple[str, list[str], Path]] = [
        (
            "integration: compileall",
            [python, "-m", "compileall", "-q", "app"],
            REPO_ROOT / "integration",
        ),
        (
            "integration: integration-status",
            [python, str(REPO_ROOT / "rpa.py"), "integration-status"],
            REPO_ROOT,
        ),
        (
            "retrieval: compileall",
            [python, "-m", "compileall", "-q", "app"],
            REPO_ROOT / "modules" / "retrieval",
        ),
        (
            "retrieval: recommend-topic (tfidf, sample corpus)",
            [
                python,
                "-m",
                "app.cli",
                "recommend-topic",
                "--query",
                "retrieval augmented generation",
                "--papers",
                "../dataset/data/processed/dev_sample.jsonl",
                "--out",
                "outputs/smoke_recommendations.json",
                "--top-k",
                "5",
                "--retrieval-strategy",
                "tfidf",
            ],
            REPO_ROOT / "modules" / "retrieval",
        ),
        (
            "pdf_nlp: build ParsedPaper from sample PDF",
            [
                python,
                "-c",
                (
                    "from pathlib import Path; "
                    "from pdf_parser import build_parsed_paper; "
                    "paper, debug=build_parsed_paper("
                    "Path('../../tests/papers/drq_v2/2107.09645v1.pdf')); "
                    "assert paper['metadata']['title']; "
                    "assert paper['sections']['abstract']; "
                    "print('pages', debug['page_count'], "
                    "'sections', len(paper['sections']), "
                    "'references', len(paper['references']))"
                ),
            ],
            REPO_ROOT / "modules" / "pdf_nlp",
        ),
    ]

    for name, command, cwd in steps[:2]:
        if not runner.command(name, command, cwd=cwd):
            return runner.summary()

    if not runner.check("dataset: canonical dev_5k.jsonl is valid", _validate_corpus):
        return runner.summary()

    for name, command, cwd in steps[2:4]:
        if not runner.command(name, command, cwd=cwd):
            return runner.summary()

    if not runner.check("retrieval: smoke output exists", _validate_retrieval_output):
        return runner.summary()

    name, command, cwd = steps[4]
    if not runner.command(name, command, cwd=cwd):
        return runner.summary()

    system_command = [python, str(REPO_ROOT / "tests" / "run_system_tests.py")]
    if args.skip_ollama:
        system_command.append("--skip-ollama")
    elif args.require_ollama:
        system_command.append("--require-ollama")
    if args.dry_run:
        system_command.append("--dry-run")
    if not runner.command(
        "tests: cross-platform system test runner",
        system_command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    ):
        return runner.summary()

    return runner.summary()


if __name__ == "__main__":
    raise SystemExit(main())
