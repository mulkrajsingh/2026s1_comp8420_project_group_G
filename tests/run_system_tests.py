"""Cross-platform system test runner for all project modules and E2E tests."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(REPO_ROOT), str(REPO_ROOT / "integration")]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _format_command(command: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _run(
    label: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    print(f"\n=== {label} ===")
    print(f"cwd: {cwd}")
    print(f"command: {_format_command(command)}")
    if dry_run:
        return 0
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            check=False,
        ).returncode
    except OSError as exc:
        print(f"Unable to run {label}: {exc}", file=sys.stderr)
        return 1


def _discover(start: str, pattern: str = "test_*.py") -> list[str]:
    return [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        start,
        "-p",
        pattern,
        "-v",
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Run deterministic PDF/retrieval E2E tests and skip Ollama E2E tests.",
    )
    group.add_argument(
        "--require-ollama",
        action="store_true",
        help="Fail when the Ollama executable is unavailable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    skip_ollama = args.skip_ollama or _enabled("SKIP_OLLAMA")
    require_ollama = args.require_ollama or _enabled("REQUIRE_OLLAMA")
    if skip_ollama and require_ollama:
        print(
            "Choose only one Ollama mode: skip or require.",
            file=sys.stderr,
        )
        return 2

    env = _pythonpath_env()
    steps: list[tuple[str, list[str], Path]] = [
        (
            "Bootstrap E2E artifacts",
            [sys.executable, "tests/harness/bootstrap_artifacts.py"],
            REPO_ROOT,
        )
    ]

    ollama_available = shutil.which("ollama") is not None
    if skip_ollama or not ollama_available:
        if require_ollama and not ollama_available:
            print("Ollama is required but was not found on PATH.", file=sys.stderr)
            return 1
        if not skip_ollama and not ollama_available:
            print("Ollama not found; running deterministic E2E tests only.")
        steps.extend(
            [
                (
                    "E2E PDF-NLP",
                    _discover("tests/e2e", "test_pdf_nlp.py"),
                    REPO_ROOT,
                ),
                (
                    "E2E retrieval",
                    _discover("tests/e2e", "test_retrieval.py"),
                    REPO_ROOT,
                ),
            ]
        )
    else:
        steps.append(
            ("All E2E tests", _discover("tests/e2e"), REPO_ROOT)
        )

    steps.extend(
        [
            (
                "Dataset unit tests",
                _discover("modules/dataset/tests"),
                REPO_ROOT,
            ),
            (
                "PDF-NLP unit tests",
                _discover("tests"),
                REPO_ROOT / "modules" / "pdf_nlp",
            ),
            (
                "Retrieval unit tests",
                _discover("tests"),
                REPO_ROOT / "modules" / "retrieval",
            ),
            (
                "LLM unit tests",
                _discover("tests"),
                REPO_ROOT / "modules" / "llm",
            ),
            (
                "Integration unit tests",
                _discover("tests"),
                REPO_ROOT / "integration",
            ),
        ]
    )

    for label, command, cwd in steps:
        code = _run(
            label,
            command,
            cwd=cwd,
            env=env,
            dry_run=args.dry_run,
        )
        if code:
            print(f"\nSystem tests failed during: {label}", file=sys.stderr)
            return code

    print("\nSystem tests complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
