"""Cross-platform repository launcher for the Research Paper Assistant.

Run from the repository root with::

    python rpa.py web
    python rpa.py analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf

The integration CLI runs from ``integration/`` so its existing output paths
remain unchanged. File arguments supplied relative to the caller's working
directory are converted to absolute paths before the working directory changes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent
INTEGRATION_ROOT = REPO_ROOT / "integration"
POSITIONAL_PATH_COMMANDS = {"analyze-pdf", "peer-review"}
EXISTING_PATH_OPTIONS = {
    "--annotations",
    "--archive",
    "--corpus",
    "--evidence",
    "--llama-cpp",
    "--manifest",
    "--paper-json",
    "--papers",
    "--raw",
    "--test-set",
}
OUTPUT_PATH_OPTIONS = {
    "--analysis-out",
    "--debug-out",
    "--enriched-paper-out",
    "--json-out",
    "--merged-dir",
    "--metadata-out",
    "--out",
    "--output-dir",
    "--report",
    "--review-out",
    "--staging",
}


def _is_absolute_path(value: str) -> bool:
    """Recognize native and Windows absolute paths on every host platform."""
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _resolve_existing_path(value: str, caller_cwd: Path) -> str:
    """Resolve a caller-relative path when it exists, otherwise preserve it."""
    if not value or _is_absolute_path(value):
        return value
    candidate = caller_cwd / value
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _resolve_output_path(value: str, caller_cwd: Path) -> str:
    """Resolve a caller-relative output path even when it does not exist yet."""
    if not value or _is_absolute_path(value):
        return value
    return str((caller_cwd / value).resolve())


def normalize_args(args: Sequence[str], caller_cwd: Path) -> list[str]:
    """Return integration CLI arguments with caller-relative paths preserved."""
    normalized = list(args)
    if (
        normalized
        and normalized[0] in POSITIONAL_PATH_COMMANDS
        and len(normalized) > 1
    ):
        normalized[1] = _resolve_existing_path(normalized[1], caller_cwd)

    index = 0
    while index < len(normalized):
        value = normalized[index]
        if value in EXISTING_PATH_OPTIONS and index + 1 < len(normalized):
            normalized[index + 1] = _resolve_existing_path(
                normalized[index + 1], caller_cwd
            )
            index += 2
            continue
        if value in OUTPUT_PATH_OPTIONS and index + 1 < len(normalized):
            normalized[index + 1] = _resolve_output_path(
                normalized[index + 1], caller_cwd
            )
            index += 2
            continue
        for option in EXISTING_PATH_OPTIONS:
            prefix = f"{option}="
            if value.startswith(prefix):
                path_value = value[len(prefix) :]
                normalized[index] = (
                    f"{prefix}{_resolve_existing_path(path_value, caller_cwd)}"
                )
                break
        else:
            for option in OUTPUT_PATH_OPTIONS:
                prefix = f"{option}="
                if value.startswith(prefix):
                    path_value = value[len(prefix) :]
                    normalized[index] = (
                        f"{prefix}{_resolve_output_path(path_value, caller_cwd)}"
                    )
                    break
        index += 1
    return normalized


def build_command(args: Sequence[str], caller_cwd: Path) -> list[str]:
    """Build the platform-neutral integration CLI command."""
    return [
        sys.executable,
        "-m",
        "app.cli",
        *normalize_args(args, caller_cwd),
    ]


def main(argv: Sequence[str] | None = None, *, caller_cwd: Path | None = None) -> int:
    """Run the integration CLI and return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    cwd = Path.cwd() if caller_cwd is None else caller_cwd
    try:
        completed = subprocess.run(
            build_command(args, cwd),
            cwd=INTEGRATION_ROOT,
            check=False,
        )
    except OSError as exc:
        print(f"Unable to start the Research Paper Assistant: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
