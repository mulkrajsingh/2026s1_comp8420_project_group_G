"""Tests for resumable Colab checkpoint handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.checkpointing import (
    atomic_copytree,
    completion_matches,
    is_valid_checkpoint,
    latest_valid_checkpoint,
    prune_checkpoints,
    require_matching_metadata,
    restore_latest_checkpoint,
    write_json_atomic,
)


def make_checkpoint(parent: Path, step: int, *, complete: bool = True) -> Path:
    checkpoint = parent / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    if complete:
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "rng_state.pth").write_bytes(b"rng")
    return checkpoint


class CheckpointingTests(unittest.TestCase):
    def test_latest_checkpoint_uses_numeric_step_and_ignores_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            make_checkpoint(parent, 50)
            make_checkpoint(parent, 100)
            make_checkpoint(parent, 900)
            make_checkpoint(parent, 1000, complete=False)
            (parent / ".checkpoint-1200.partial").mkdir()

            latest = latest_valid_checkpoint(parent)

            self.assertIsNotNone(latest)
            self.assertEqual(latest.name, "checkpoint-900")

    def test_valid_checkpoint_requires_full_trainer_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            valid = make_checkpoint(parent, 50)
            incomplete = make_checkpoint(parent, 100, complete=False)

            self.assertTrue(is_valid_checkpoint(valid))
            self.assertFalse(is_valid_checkpoint(incomplete))

    def test_atomic_copy_and_restore_latest_checkpoint(self) -> None:
        metadata = {"dataset_rows": 17298, "split_seed": 42}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drive_run = root / "drive"
            local_run = root / "local"
            drive_run.mkdir()
            make_checkpoint(drive_run, 50)
            make_checkpoint(drive_run, 150)
            write_json_atomic(drive_run / "run_metadata.json", metadata)

            restored = restore_latest_checkpoint(
                drive_run,
                local_run,
                metadata,
            )

            self.assertIsNotNone(restored)
            self.assertEqual(restored.name, "checkpoint-150")
            self.assertTrue(is_valid_checkpoint(restored))
            trainer_state = json.loads(
                (restored / "trainer_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(trainer_state["global_step"], 150)

    def test_metadata_mismatch_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metadata.json"
            write_json_atomic(path, {"dataset_rows": 16998, "split_seed": 42})

            with self.assertRaisesRegex(ValueError, "dataset_rows"):
                require_matching_metadata(
                    path,
                    {"dataset_rows": 17298, "split_seed": 42},
                )

    def test_retention_keeps_latest_three_valid_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            for step in (50, 100, 150, 200):
                make_checkpoint(parent, step)

            removed = prune_checkpoints(parent, keep=3)

            self.assertEqual([path.name for path in removed], ["checkpoint-50"])
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["checkpoint-100", "checkpoint-150", "checkpoint-200"],
            )

    def test_completion_marker_must_match_run(self) -> None:
        metadata = {"dataset_sha256": "abc", "max_seq_length": 4096}
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "completed.json"
            write_json_atomic(marker, metadata)

            self.assertTrue(completion_matches(marker, metadata))
            self.assertFalse(
                completion_matches(
                    marker,
                    {"dataset_sha256": "different", "max_seq_length": 4096},
                )
            )

    def test_atomic_copy_replaces_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            (destination / "old.txt").write_text("old", encoding="utf-8")

            atomic_copytree(source, destination)

            self.assertTrue((destination / "new.txt").is_file())
            self.assertFalse((destination / "old.txt").exists())


if __name__ == "__main__":
    unittest.main()
