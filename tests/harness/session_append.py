"""Append JSONL events when COMP8420_SESSION_LOG is set (module CLI integration)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping


def append_session_event(event: str, payload: Mapping[str, Any]) -> None:
    path = os.environ.get("COMP8420_SESSION_LOG")
    if not path:
        return
    line = json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": dict(payload),
        },
        ensure_ascii=True,
    ) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
