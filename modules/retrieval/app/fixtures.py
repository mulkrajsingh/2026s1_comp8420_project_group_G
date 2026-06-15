"""PaperRecord corpus paths and evaluation query fixtures.

Loads the shared dev corpus, keyword-style benchmark queries, and optional
user-like queries with hand-labelled relevant paper IDs.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import read_jsonl

_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
_DEFAULT_JSONL = _DATA_DIR / "dev_5k.jsonl"
_REPO_CORPUS = (
    Path(__file__).resolve().parents[3]
    / "modules"
    / "dataset"
    / "data"
    / "processed"
    / "dev_5k.jsonl"
)


def default_corpus_path() -> Path:
    """Return the canonical production corpus path."""
    if _REPO_CORPUS.is_file():
        return _REPO_CORPUS
    if _DEFAULT_JSONL.is_file():
        return _DEFAULT_JSONL
    raise FileNotFoundError(
        f"No corpus found. Expected {_REPO_CORPUS} or {_DEFAULT_JSONL}."
    )


def load_papers(path: str | Path) -> list[dict]:
    """Load PaperRecord list from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Corpus file not found: {p}")
    return read_jsonl(p)


# Canonical evaluation queries with hand-labelled relevant paper_ids
EVAL_QUERIES = [
    {
        "query": "natural language processing text classification",
        "relevant_ids": [
            "1004.3183", "1108.3848", "1106.0411", "1105.4318", "1203.2498",
        ],
    },
    {
        "query": "reinforcement learning policy reward agent",
        "relevant_ids": [
            "1104.5687", "1107.0048", "1009.2566", "1012.1552", "1205.4839",
        ],
    },
    {
        "query": "support vector machine kernel classification",
        "relevant_ids": [
            "1101.2987", "0804.0188", "0912.0874", "0906.4391", "1010.0535",
        ],
    },
    {
        "query": "neural network deep learning representation",
        "relevant_ids": [
            "1202.2770", "1111.4930", "0804.3269", "1105.0972", "0912.1007",
        ],
    },
    {
        "query": "bayesian probabilistic graphical model",
        "relevant_ids": [
            "1111.6925", "0706.2040", "1106.1799", "1011.0935", "1004.2304",
        ],
    },
]

USER_EVAL_PATH = _DATA_DIR / "eval_queries_user.json"


def load_user_eval_queries(path: Path | None = None) -> list[dict]:
    """Load natural-language evaluation queries with manual gold labels."""
    query_path = path or USER_EVAL_PATH
    if not query_path.is_file():
        raise FileNotFoundError(f"User evaluation queries missing: {query_path}")
    import json

    payload = json.loads(query_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {query_path}")
    queries = []
    for row in payload:
        queries.append(
            {
                "query": row["query"],
                "relevant_ids": list(row["relevant_ids"]),
            }
        )
    return queries


def load_eval_queries(query_set: str = "all") -> list[dict]:
    """Return keyword, user-like, or combined evaluation queries."""
    normalized = query_set.strip().lower()
    if normalized == "keyword":
        return list(EVAL_QUERIES)
    if normalized == "user":
        return load_user_eval_queries()
    if normalized == "all":
        return list(EVAL_QUERIES) + load_user_eval_queries()
    raise ValueError(
        f"Unsupported query set {query_set!r}; expected keyword, user, or all."
    )
