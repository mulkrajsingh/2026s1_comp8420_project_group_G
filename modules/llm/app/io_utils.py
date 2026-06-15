"""Small JSON and text helpers used by the LLM CLI and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: Path) -> None:
    """Create parent directories for a file path when missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    """Write pretty-printed JSON to disk."""
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read and parse one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk."""
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write objects to JSONL, one record per line."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
