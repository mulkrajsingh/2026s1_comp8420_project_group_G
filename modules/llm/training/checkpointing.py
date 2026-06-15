"""Resumable Hugging Face checkpoint helpers for Colab training."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_CHECKPOINT_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)
MODEL_FILE_PATTERNS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
RNG_FILE_PATTERNS = ("rng_state.pth", "rng_state_*.pth")


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def is_valid_checkpoint(path: Path) -> bool:
    if not path.is_dir() or checkpoint_step(path) is None:
        return False
    if any(not (path / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
        return False
    if not any((path / name).is_file() for name in MODEL_FILE_PATTERNS):
        return False
    return any(any(path.glob(pattern)) for pattern in RNG_FILE_PATTERNS)


def valid_checkpoints(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    checkpoints = [
        path for path in parent.iterdir()
        if is_valid_checkpoint(path)
    ]
    return sorted(checkpoints, key=lambda path: checkpoint_step(path) or -1)


def latest_valid_checkpoint(parent: Path) -> Path | None:
    checkpoints = valid_checkpoints(parent)
    return checkpoints[-1] if checkpoints else None


def atomic_copytree(source: Path, destination: Path) -> Path:
    """Copy a directory without exposing a partial destination."""
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    shutil.copytree(source, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination


def prune_checkpoints(parent: Path, keep: int) -> list[Path]:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    checkpoints = valid_checkpoints(parent)
    removed: list[Path] = []
    for checkpoint in checkpoints[:-keep]:
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return removed


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def metadata_mismatches(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: checkpoint={actual_value!r}, current={expected_value!r}"
            )
    return mismatches


def require_matching_metadata(
    metadata_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    actual = load_json(metadata_path)
    mismatches = metadata_mismatches(actual, expected)
    if mismatches:
        raise ValueError(
            "Checkpoint metadata does not match the current run:\n- "
            + "\n- ".join(mismatches)
        )
    return actual


def restore_latest_checkpoint(
    drive_run_dir: Path,
    local_output_dir: Path,
    expected_metadata: dict[str, Any],
) -> Path | None:
    latest = latest_valid_checkpoint(drive_run_dir)
    if latest is None:
        return None
    require_matching_metadata(
        drive_run_dir / "run_metadata.json",
        expected_metadata,
    )
    destination = local_output_dir / latest.name
    atomic_copytree(latest, destination)
    if not is_valid_checkpoint(destination):
        raise RuntimeError(f"Restored checkpoint is incomplete: {destination}")
    return destination


def completion_matches(
    completion_path: Path,
    expected_metadata: dict[str, Any],
) -> bool:
    if not completion_path.is_file():
        return False
    actual = load_json(completion_path)
    return not metadata_mismatches(actual, expected_metadata)
