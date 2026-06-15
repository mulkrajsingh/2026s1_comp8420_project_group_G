"""Helpers to load Hugging Face datasets for the open-data notebook."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any


def load_qasper(split: str = "train") -> Any:
    from datasets import load_dataset

    return load_dataset("allenai/qasper", split=split)


def load_scitldr(split: str = "train", config: str = "Abstract") -> Any:
    from datasets import load_dataset

    return load_dataset("allenai/scitldr", config, split=split)


def load_scicite(split: str = "train", config: str = "extended") -> Any:
    from datasets import load_dataset

    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    return load_dataset("allenai/scicite", config, split=split, trust_remote_code=True)


def load_peerread(split: str = "train", config: str = "reviews") -> Any:
    from datasets import load_dataset

    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    return load_dataset("allenai/peer_read", config, split=split, trust_remote_code=True)


def iter_researchqa_jsonl(split: str = "test") -> Iterator[dict[str, Any]]:
    """Stream ResearchQA from Hub JSONL (avoids datasets CastError on mixed columns)."""
    if split not in {"test", "eval"}:
        raise ValueError(
            f"Unsupported ResearchQA split {split!r}. Hub only ships eval data as eval_dataset.jsonl (use 'test')."
        )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="khoj-ai/ResearchQA",
        filename="eval_dataset.jsonl",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def load_researchqa(split: str = "test") -> Iterator[dict[str, Any]]:
    """Load ResearchQA rows for the hybrid open-data notebook.

    ``load_dataset`` without streaming fails on this Hub repo (mixed JSONL columns).
    Reading ``eval_dataset.jsonl`` directly avoids DatasetGenerationCastError.
    """
    return iter_researchqa_jsonl(split=split)
