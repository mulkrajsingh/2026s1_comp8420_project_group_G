"""Test run logger — session JSONL + artifact capture under tests/logs/."""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import INTEGRATION_ROOT, LLM_ROOT, LOGS_DIR, PDF_NLP_ROOT, RETRIEVAL_ROOT


@dataclass
class TestRunLogger:
    """One test scenario: session file, env for subprocesses, artifact manifest."""

    run_id: str
    log_dir: Path
    session_path: Path
    steps: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default=0.0, repr=False)

    @classmethod
    def create(cls, name: str) -> TestRunLogger:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}_{name}"
        log_dir = LOGS_DIR / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        session_path = log_dir / "session.jsonl"
        session_path.write_text("", encoding="utf-8")
        logger = cls(run_id=run_id, log_dir=log_dir, session_path=session_path)
        logger._t0 = time.time()
        logger.record_step("run_start", {"name": name})
        return logger

    def session_env(self) -> dict[str, str]:
        return {"COMP8420_SESSION_LOG": str(self.session_path)}

    def record_step(self, step: str, detail: dict[str, Any] | None = None) -> None:
        elapsed_ms = round((time.time() - self._t0) * 1000, 1)
        entry = {"step": step, "elapsed_ms": elapsed_ms}
        if detail:
            entry.update(detail)
        self.steps.append(entry)

    def capture_artifacts(self) -> list[str]:
        """Copy module/integration outputs into this run's artifacts folder."""
        artifacts_dir = self.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for label, root in (
            ("integration", INTEGRATION_ROOT / "outputs"),
            ("pdf_nlp", PDF_NLP_ROOT / "outputs"),
            ("retrieval", RETRIEVAL_ROOT / "outputs"),
            ("llm", LLM_ROOT / "outputs"),
        ):
            if not root.is_dir():
                continue
            dest = artifacts_dir / label
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(root, dest)
            copied.append(str(dest))
        self.record_step("artifacts_captured", {"paths": copied})
        return copied

    def finalize(self, *, exit_code: int = 0) -> Path:
        manifest = {
            "run_id": self.run_id,
            "session_log": str(self.session_path),
            "steps": self.steps,
            "exit_code": exit_code,
        }
        manifest_path = self.log_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def __enter__(self) -> TestRunLogger:
        os.environ["COMP8420_SESSION_LOG"] = str(self.session_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.record_step("run_error", {"error": str(exc)})
        self.capture_artifacts()
        self.finalize(exit_code=0 if exc_type is None else 1)
        os.environ.pop("COMP8420_SESSION_LOG", None)
