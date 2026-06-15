"""Single-worker local job queue with cursor-based structured progress events.

The queue intentionally serialises expensive PDF-NLP and Ollama work so local
machines do not load multiple large models concurrently. Job state is held in
memory; durable execution evidence remains in the structured session directory.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .ollama_runtime import MODEL_MANAGER
from .providers.live_providers import _integration_root
from .session_log import new_timestamp_id, refresh_session_artifacts

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="comp8420-job")
_LOCK = threading.Lock()
_JOBS: dict[str, "JobRecord"] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    """Mutable state for one queued local analysis."""

    job_id: str
    kind: str
    state: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def run_dir(self) -> Path:
        """Return the durable session directory for this job."""
        return _integration_root() / "data" / "sessions" / self.job_id

    @property
    def session_path(self) -> Path:
        """Return the shared JSONL event stream for this job."""
        return self.run_dir / "session.jsonl"


def _event(
    record: JobRecord,
    *,
    status: str,
    message: str,
    error: str | None = None,
) -> None:
    record.run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _now()
    row = {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "ts": timestamp,
        "run_id": record.job_id,
        "event_id": f"{record.job_id}-job-{uuid4().hex[:8]}",
        "event": "job_state",
        "component": "integration",
        "phase": "job",
        "status": status,
        "source": "live",
        "message": message,
        "duration_ms": None,
        "metrics": {},
        "artifacts": [],
        "error": error,
        "payload": {"kind": record.kind, "state": status},
    }
    with record.session_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _new_record(kind: str) -> JobRecord:
    job_id = new_timestamp_id()
    record = JobRecord(job_id=job_id, kind=kind)
    with _LOCK:
        _JOBS[job_id] = record
    _event(record, status="queued", message=f"{kind} job queued")
    return record


def _run_job(
    record: JobRecord,
    function: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    *,
    cleanup_path: str | None = None,
) -> None:
    with _LOCK:
        record.state = "running"
        record.started_at = _now()
    _event(record, status="running", message=f"{record.kind} job running")
    try:
        with MODEL_MANAGER.use(
            backend="ollama",
            model=str(kwargs.get("llm_model", "qwen3:8b")),
        ):
            output = function(run_id=record.job_id, **kwargs)
        result = output.get("result")
        serialised = result.to_dict() if hasattr(result, "to_dict") else result
        with _LOCK:
            record.result = serialised
            record.state = "succeeded"
            record.completed_at = _now()
        _event(record, status="succeeded", message=f"{record.kind} job succeeded")
    except Exception as exc:
        with _LOCK:
            record.error = str(exc)
            record.state = "failed"
            record.completed_at = _now()
        _event(
            record,
            status="failed",
            message=f"{record.kind} job failed",
            error=str(exc),
        )
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            os.remove(cleanup_path)
        refresh_session_artifacts(
            session_path=record.session_path,
            run_id=record.job_id,
            run_dir=record.run_dir,
            started_at=record.created_at,
        )


def submit_pdf_job(pdf_path: str, options: dict[str, Any]) -> JobRecord:
    """Queue a PDF analysis and delete its temporary upload afterward."""
    from .service import run_analyze_pdf

    record = _new_record("analyze-pdf")
    _EXECUTOR.submit(
        _run_job,
        record,
        run_analyze_pdf,
        {"pdf_path": pdf_path, **options},
        cleanup_path=pdf_path,
    )
    return record


def submit_topic_job(query: str, options: dict[str, Any]) -> JobRecord:
    """Queue a topic analysis."""
    from .service import run_topic

    record = _new_record("search-topic")
    _EXECUTOR.submit(
        _run_job,
        record,
        run_topic,
        {"query": query, **options},
    )
    return record


def get_job(job_id: str, *, after: int = 0) -> dict[str, Any]:
    """Return current state and session events after a zero-based cursor."""
    with _LOCK:
        record = _JOBS.get(job_id)
    if record is None:
        raise KeyError(job_id)
    events: list[dict[str, Any]] = []
    if record.session_path.is_file():
        for line in record.session_path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    safe_after = max(0, min(after, len(events)))
    return {
        "job_id": record.job_id,
        "kind": record.kind,
        "state": record.state,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "events": events[safe_after:],
        "cursor": len(events),
        "result": record.result if record.state == "succeeded" else None,
        "error": record.error,
    }
