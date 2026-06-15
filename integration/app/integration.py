"""Report whether required member-module entry points are present on disk."""

from pathlib import Path

from .io_paths import write_text

REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifacts():
    return [
        (
            "paper_source",
            REPO_ROOT / "modules/dataset/data/processed/dev_5k_balanced.jsonl",
            "modules/dataset",
            "file-backed",
        ),
        (
            "parser",
            REPO_ROOT / "modules/pdf_nlp/app/cli.py",
            "modules/pdf_nlp",
            "live",
        ),
        (
            "recommender",
            REPO_ROOT / "modules/retrieval/app/cli.py",
            "modules/retrieval",
            "live",
        ),
        (
            "synthesizer",
            REPO_ROOT / "modules/llm/app/cli.py",
            "modules/llm",
            "live",
        ),
    ]


def write_integration_status() -> str:
    """Generate a status table without mutating runtime provider state."""
    lines = [
        "# Integration status",
        "",
        "| Role | Module | Required path | Present | Production source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for role, path, module, source in _artifacts():
        lines.append(
            f"| {role} | {module} | `{path}` | "
            f"{'yes' if path.exists() else 'no'} | {source} |"
        )
    lines.extend(
        [
            "",
            "Production requests construct providers per request; this command "
            "does not register global state.",
            "",
        ]
    )
    return write_text("integration_status.md", "\n".join(lines))
