"""Subprocess wrappers for module and integration CLIs."""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Mapping, Sequence

from .paths import INTEGRATION_ROOT, LLM_ROOT, PDF_NLP_ROOT, RETRIEVAL_ROOT


def _run(
    cwd: Path,
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    merged = dict(__import__("os").environ)
    if env:
        merged.update(env)
    interpreter = sys.executable
    if cwd == PDF_NLP_ROOT:
        interpreter = os.environ.get("COMP8420_PDF_NLP_PYTHON", interpreter)
    return subprocess.run(
        [interpreter, "-m", "app.cli", *args],
        cwd=str(cwd),
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_parse_pdf(
    pdf: Path,
    out: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        PDF_NLP_ROOT,
        [
            "parse-pdf",
            "--pdf",
            str(pdf),
            "--out",
            str(out),
            "--debug-out",
            str(out.parent / f"{out.stem}_debug.json"),
        ],
        env=env,
    )


def run_analyze_paper(
    pdf: Path,
    out: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the full production PDF-NLP command on one real paper."""
    out.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        PDF_NLP_ROOT,
        [
            "analyze-paper",
            "--pdf",
            str(pdf),
            "--out",
            str(out),
            "--debug-out",
            str(out.parent / f"{out.stem}_debug.json"),
            "--analysis-out",
            str(out.parent / f"{out.stem}_analysis.json"),
            "--review-out",
            str(out.parent / f"{out.stem}_review.json"),
        ],
        env=env,
        timeout=900,
    )


def run_recommend_topic(
    query: str,
    papers: Path,
    out: Path,
    *,
    top_k: int = 10,
    embedding_model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "recommend-topic",
        "--query",
        query,
        "--papers",
        str(papers),
        "--out",
        str(out),
        "--top-k",
        str(top_k),
    ]
    if embedding_model:
        args.extend(["--embedding-model", embedding_model])
    return _run(RETRIEVAL_ROOT, args, env=env, timeout=900)


def run_llm_summarize(
    paper_json: Path,
    out: Path,
    *,
    query: str = "Summarize the supplied paper.",
    model: str = "qwen3:8b",
    style: str = "technical",
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        LLM_ROOT,
        [
            "summarize",
            "--paper",
            str(paper_json),
            "--query",
            query,
            "--out",
            str(out),
            "--model",
            model,
            "--style",
            style,
        ],
        env=env,
        timeout=900,
    )


def run_llm_synthesize(
    evidence: Path,
    out: Path,
    *,
    model: str = "qwen3:8b",
    style: str = "technical",
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        LLM_ROOT,
        [
            "synthesize",
            "--evidence",
            str(evidence),
            "--out",
            str(out),
            "--json-out",
            str(out.parent / "analysis_result_from_llm.json"),
            "--model",
            model,
            "--style",
            style,
        ],
        env=env,
        timeout=900,
    )


def run_integration(
    command: str,
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    return _run(INTEGRATION_ROOT, [command, *args], env=env, timeout=timeout)
