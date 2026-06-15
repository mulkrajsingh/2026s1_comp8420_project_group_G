"""Append structured JSONL events for the PDF-NLP subprocess.

When ``COMP8420_SESSION_LOG`` is set, writes schema-versioned events with
optional redaction of configured secret substrings.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def _redact(value: Any) -> Any:
    raw = os.environ.get("COMP8420_REDACT_VALUES", "[]")
    try:
        redactions = [str(item) for item in json.loads(raw)]
    except (TypeError, json.JSONDecodeError):
        redactions = []
    if isinstance(value, str):
        for item in redactions:
            if item:
                value = value.replace(item, "[redacted-upload]")
        return value
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    return value


def _verbose_session_log() -> bool:
    return os.environ.get("COMP8420_VERBOSE_SESSION_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def append_session_event(
    event: str,
    payload: Mapping[str, Any],
    *,
    phase: str | None = None,
    status: str | None = None,
    source: str = "live",
    message: str = "",
    duration_ms: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Append one schema-versioned event when a session path is configured."""
    path = os.environ.get("COMP8420_SESSION_LOG")
    if not path:
        return
    inferred_status = status or (
        "failed" if error else "completed" if event.endswith(("complete", "output")) else "info"
    )
    clean_payload = _redact(dict(payload))
    line = json.dumps(
        {
            "schema_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": os.environ.get("COMP8420_RUN_ID", "standalone"),
            "turn_id": os.environ.get("COMP8420_TURN_ID"),
            "event_id": f"pdf-{uuid4().hex}",
            "event": event,
            "component": "pdf_nlp",
            "phase": phase or event,
            "status": inferred_status,
            "source": source,
            "message": message or event.replace("_", " "),
            "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
            "metrics": _redact(dict(metrics or {})),
            "artifacts": _redact(artifacts or []),
            "error": _redact(error),
            "payload": clean_payload,
        },
        ensure_ascii=True,
    ) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
