"""Merge hybrid open data, project arXiv RAG, and optional seed rows into final train JSONL."""

from __future__ import annotations

import random
import shutil
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from lora_dataset.io import read_jsonl, validate_messages_record, write_jsonl
from lora_dataset.paths import (
    FINAL_DATASET_DIR,
    HYBRID_JSONL,
    PAPER_ONLY_JSONL,
    RAG_JSONL,
    SEED,
    TRAIN_JSONL,
    TRAIN_MANIFEST,
    TRAIN_ZIP,
    WORKSTREAM_ROOT,
)
from lora_dataset.seeds import build_seed_rows


def _write_train_zip() -> Path:
    """Zip the merged JSONL for upload to Colab or Drive."""
    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_ZIP.is_file():
        TRAIN_ZIP.unlink()
    archive_path = shutil.make_archive(
        str(TRAIN_ZIP.with_suffix("")),
        "zip",
        root_dir=TRAIN_JSONL.parent,
        base_dir=TRAIN_JSONL.name,
    )
    return Path(archive_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_train(*, include_seeds: bool = True, shuffle_seed: int = SEED) -> dict[str, int]:
    """Merge dataset shards, shuffle, write train JSONL, zip, and manifest."""
    merged: list[dict[str, Any]] = []
    inputs: list[str] = []

    if include_seeds:
        merged.extend(build_seed_rows())
        inputs.append("local_train_anchor (6)")

    for label, path in [
        ("hybrid_open_data", HYBRID_JSONL),
        ("project_arxiv_rag", RAG_JSONL),
        ("project_paper_only", PAPER_ONLY_JSONL),
    ]:
        if not path.is_file():
            raise SystemExit(f"Missing input for merge: {path}")
        rows = read_jsonl(path)
        merged.extend(rows)
        inputs.append(f"{label}: {path.name} ({len(rows)} rows)")

    for record in merged:
        validate_messages_record(record)

    random.Random(shuffle_seed).shuffle(merged)
    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_JSONL, merged)
    zip_path = _write_train_zip()

    counts = Counter(str(row.get("source", "unknown")) for row in merged)
    task_counts = Counter(str(row.get("task", "unknown")) for row in merged)
    train_sha256 = _sha256(TRAIN_JSONL)
    zip_sha256 = _sha256(zip_path)
    TRAIN_MANIFEST.write_text(
        "\n".join(
            [
                "# Research LoRA Train, Hybrid plus arXiv RAG Manifest",
                "",
                f"Total training rows: {len(merged)}",
                f"Output JSONL: `{TRAIN_JSONL.relative_to(WORKSTREAM_ROOT)}`",
                f"Output zip: `{zip_path.relative_to(WORKSTREAM_ROOT)}`",
                f"Shuffle seed: {shuffle_seed}",
                "",
                "## Inputs",
                *[f"- {line}" for line in inputs],
                "",
                "## Rows by source",
                *[f"- `{source}`: {count}" for source, count in sorted(counts.items())],
                "",
                "## Rows by task",
                *[f"- `{task}`: {count}" for task, count in sorted(task_counts.items())],
                "",
                "## Checksums",
                f"- Train JSONL SHA256: `{train_sha256}`",
                f"- Train zip SHA256: `{zip_sha256}`",
                "",
                "Colab: upload the zip from final_dataset to Google Drive, then set TRAIN_JSONL in train_lora_adapter.ipynb.",
                "Train: notebooks/train_lora_adapter.ipynb (Colab or local Jupyter).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(merged)} merged rows to {TRAIN_JSONL}")
    print(f"Wrote train zip: {zip_path}")
    return dict(counts)
