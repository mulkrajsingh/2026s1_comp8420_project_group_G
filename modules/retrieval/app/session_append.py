"""Append schema-versioned retrieval events to the active integration session."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def _redact(value: Any) -> Any:
    try:
        redactions = json.loads(os.environ.get("COMP8420_REDACT_VALUES", "[]"))
    except json.JSONDecodeError:
        redactions = []
    if isinstance(value, str):
        for item in redactions:
            value = value.replace(str(item), "[redacted-upload]")
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


def append_session_event(event: str, payload: Mapping[str, Any]) -> None:
    """Append one retrieval event while retaining the legacy event name."""
    if event in {"user_input", "user_output"} and not _verbose_session_log():
        return
    path = os.environ.get("COMP8420_SESSION_LOG")
    if not path:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    status = (
        "started"
        if event == "user_input"
        else "failed"
        if payload.get("error")
        else "completed"
    )
    phase = "retrieval" if event in {"retrieval", "user_input", "user_output"} else event
    metrics = {}
    if event == "retrieval":
        metrics["result_count"] = len(payload.get("top_papers") or [])
    line = json.dumps(
        {
            "schema_version": "1.0",
            "timestamp": timestamp,
            "run_id": os.environ.get("COMP8420_RUN_ID", "standalone"),
            "turn_id": os.environ.get("COMP8420_TURN_ID"),
            "event_id": f"retrieval-{uuid4().hex}",
            "event": event,
            "component": "retrieval",
            "phase": phase,
            "status": status,
            "source": "live",
            "message": event.replace("_", " "),
            "duration_ms": payload.get("duration_ms"),
            "metrics": metrics,
            "artifacts": _redact(payload.get("artifact_paths") or []),
            "error": _redact(payload.get("error")),
            "payload": _redact(dict(payload)),
        },
        ensure_ascii=True,
    ) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
