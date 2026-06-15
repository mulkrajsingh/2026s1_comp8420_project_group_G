"""Shared paths for system tests — importable from any module or integration folder."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
PAPERS_DIR = TESTS_DIR / "papers"
PROMPTS_DIR = TESTS_DIR / "prompts"
LOGS_DIR = TESTS_DIR / "logs"
FIXTURE_ARTIFACTS_DIR = PAPERS_DIR / "artifacts"
# E2E commands write here so committed evidence remains unchanged.
ARTIFACTS_DIR = LOGS_DIR / "artifacts"

CORPUS_PATH = REPO_ROOT / "modules" / "dataset" / "data" / "processed" / "dev_5k.jsonl"
CORPUS_SAMPLE_PATH = (
    REPO_ROOT / "modules" / "dataset" / "data" / "processed" / "dev_sample.jsonl"
)

INTEGRATION_ROOT = REPO_ROOT / "integration"
PDF_NLP_ROOT = REPO_ROOT / "modules" / "pdf_nlp"
RETRIEVAL_ROOT = REPO_ROOT / "modules" / "retrieval"
LLM_ROOT = REPO_ROOT / "modules" / "llm"

PAPER_PDFS = {
    "drq_v2": PAPERS_DIR / "drq_v2" / "2107.09645v1.pdf",
    "siga": PAPERS_DIR / "siga" / "SIGA_Self_Evolving_Coding_Agent_Adapters_for_Scientific_Simulation.pdf",
    "transformer": PAPERS_DIR / "transformer" / "1706.03762v7.pdf",
    "bert": PAPERS_DIR / "bert" / "1810.04805v2.pdf",
    "rag": PAPERS_DIR / "rag" / "2005.11401v4.pdf",
}


def paper_path(name: str) -> Path:
    """Return path to a named test paper PDF."""
    if name not in PAPER_PDFS:
        raise KeyError(f"Unknown paper {name!r}; known: {sorted(PAPER_PDFS)}")
    path = PAPER_PDFS[name]
    if not path.is_file():
        raise FileNotFoundError(f"Test paper missing: {path}")
    return path


def corpus_path(*, sample: bool = False) -> Path:
    path = CORPUS_SAMPLE_PATH if sample else CORPUS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Corpus missing: {path}")
    return path


def parsed_artifact_path(paper: str) -> Path:
    return ARTIFACTS_DIR / f"{paper}_parsed.json"


def rag_pack_artifact_path(paper: str = "drq_v2") -> Path:
    return ARTIFACTS_DIR / f"{paper}_rag_pack.json"
