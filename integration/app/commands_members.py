"""Member-module commands registered on the shared integration CLI.

Each Typer command shells out to the owning module CLI or dataset script so
reviewers can exercise member work from one entry point under ``integration/``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from .io_paths import write_text

_REPO_ROOT = Path(__file__).resolve().parents[2]

MODULE_PATHS = {
    "dataset": _REPO_ROOT / "modules" / "dataset",
    "pdf_nlp": _REPO_ROOT / "modules" / "pdf_nlp",
    "retrieval": _REPO_ROOT / "modules" / "retrieval",
    "llm": _REPO_ROOT / "modules" / "llm",
}

MEMBER_COMMANDS = {
    "prepare-subset": (
        "dataset",
        "Filter + sample the arXiv corpus to dev_5k_balanced.jsonl",
    ),
    "run-eda": (
        "dataset",
        "EDA charts + dataset stats (notebook)",
    ),
    "classify-domains": (
        "dataset",
        "TF-IDF + Logistic Regression domain classifier",
    ),
    "parse-pdf": (
        "pdf_nlp",
        "PDF -> ParsedPaper (sections, refs)",
    ),
    "basic-nlp": (
        "pdf_nlp",
        "POS, NER, keyphrases, extractive summary",
    ),
    "peer-review-checks": (
        "pdf_nlp",
        "Structural peer-review checks",
    ),
    "build-retrieval-index": (
        "retrieval",
        "Build TF-IDF/BM25/SPECTER2 index",
    ),
    "recommend-topic": (
        "retrieval",
        "Hybrid ranker -> Recommendation",
    ),
    "evaluate-retrieval": (
        "retrieval",
        "P@K, Recall@K, MRR, nDCG",
    ),
    "synthesize": (
        "llm",
        "LLM summary/findings/gaps (RAG-grounded)",
    ),
    "compare-prompts": (
        "llm",
        "zero-shot vs few-shot vs CoT vs ReAct",
    ),
    "compare-models": (
        "llm",
        "base vs LoRA model comparison",
    ),
}

_DATASET_SCRIPT_COMMANDS = {
    "prepare-subset": "scripts/build_balanced_corpus.py",
    "classify-domains": "scripts/evaluate_domain_classifier.py",
}

_MODULE_CLI_COMMANDS = {
    "parse-pdf": ("pdf_nlp", "parse-pdf"),
    "basic-nlp": ("pdf_nlp", "basic-nlp"),
    "peer-review-checks": ("pdf_nlp", "peer-review-checks"),
    "build-retrieval-index": ("retrieval", "build-retrieval-index"),
    "recommend-topic": ("retrieval", "recommend-topic"),
    "evaluate-retrieval": ("retrieval", "evaluate-retrieval"),
    "synthesize": ("llm", "synthesize"),
    "compare-prompts": ("llm", "compare-prompts"),
    "compare-models": ("llm", "compare-models"),
}


def _repo_root() -> Path:
    return _REPO_ROOT


def _forward_args(ctx: typer.Context) -> list[str]:
    return list(ctx.args)


def _run_process(command: list[str], *, cwd: Path) -> int:
    proc = subprocess.run(command, cwd=str(cwd))
    return proc.returncode


def _delegate_module_cli(module_key: str, module_command: str, ctx: typer.Context) -> None:
    module_dir = MODULE_PATHS[module_key]
    command = [
        sys.executable,
        "-m",
        "app.cli",
        module_command,
        *_forward_args(ctx),
    ]
    code = _run_process(command, cwd=module_dir)
    raise typer.Exit(code)


def _delegate_dataset_script(script_rel: str, ctx: typer.Context) -> None:
    module_dir = MODULE_PATHS["dataset"]
    script_path = module_dir / script_rel
    if not script_path.is_file():
        typer.echo(f"Dataset script missing: {script_path}", err=True)
        raise typer.Exit(1)
    command = [sys.executable, str(script_path), *_forward_args(ctx)]
    code = _run_process(command, cwd=module_dir)
    raise typer.Exit(code)


def register(app: typer.Typer):
    """Attach delegated member commands to the integration Typer app."""

    def _make(command_name: str, workstream: str, description: str):
        def _cmd(ctx: typer.Context):
            if command_name == "run-eda":
                notebook = _repo_root() / "modules" / "dataset" / "03_eda_1.ipynb"
                typer.echo(
                    "EDA is implemented in the dataset notebook "
                    f"{notebook.relative_to(_repo_root())}. "
                    "Run it in Jupyter to regenerate figures under "
                    "modules/dataset/results/eda/."
                )
                raise typer.Exit(0)

            if command_name in _DATASET_SCRIPT_COMMANDS:
                _delegate_dataset_script(_DATASET_SCRIPT_COMMANDS[command_name], ctx)
                return

            module_key, module_command = _MODULE_CLI_COMMANDS[command_name]
            _delegate_module_cli(module_key, module_command, ctx)

        _cmd.__doc__ = f"[{workstream}] {description}"
        app.command(
            command_name,
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        )(_cmd)

    for name, (workstream, desc) in MEMBER_COMMANDS.items():
        _make(name, workstream, desc)


def write_command_matrix() -> str:
    """Generate outputs/cli_command_matrix.md."""
    integration_commands = [
        ("analyze-pdf", "Analyze an uploaded PDF -> AnalysisResult"),
        ("search-topic", "Topic/text search -> AnalysisResult"),
        ("recommend", "Recommend related papers for a parsed paper"),
        ("peer-review", "Peer-review assistance for a PDF"),
        ("chat", "Follow-up question, optionally grounded"),
        ("run", "Topic search with real corpus + Ollama + session log"),
        ("web", "Build and serve the Vite UI and FastAPI backend"),
    ]
    lines = [
        "# CLI command matrix",
        "",
        "Every command runs through `python -m app.cli` from `integration/`.",
        "Member commands delegate to the owning module CLI or dataset script.",
        "",
        "## Integration commands",
        "",
        "| Command | Purpose |",
        "| --- | --- |",
    ]
    for name, desc in integration_commands:
        lines.append(f"| `{name}` | {desc} |")
    lines += [
        "",
        "## Member commands",
        "",
        "| Command | Workstream | Purpose | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name, (workstream, desc) in MEMBER_COMMANDS.items():
        status = "notebook pointer" if name == "run-eda" else "delegated"
        lines.append(f"| `{name}` | {workstream} | {desc} | {status} |")
    lines.append("")
    return write_text("cli_command_matrix.md", "\n".join(lines))
