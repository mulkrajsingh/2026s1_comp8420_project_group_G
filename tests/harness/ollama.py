"""Ollama availability check for system tests."""
from __future__ import annotations

import os
import shutil
import subprocess

DEFAULT_MODEL = os.environ.get("COMP8420_OLLAMA_MODEL", "qwen3:8b")


def ollama_available(model: str = DEFAULT_MODEL) -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    base = model.split(":")[0]
    return model in proc.stdout or base in proc.stdout


def require_ollama_or_skip(model: str = DEFAULT_MODEL) -> None:
    """Raise unittest.SkipTest when Ollama or the model is unavailable."""
    import unittest

    if shutil.which("ollama") is None:
        raise unittest.SkipTest("Ollama not installed — required for system E2E tests")
    if not ollama_available(model):
        raise unittest.SkipTest(
            f"Ollama model {model!r} not available. Run: ollama pull {model}"
        )
