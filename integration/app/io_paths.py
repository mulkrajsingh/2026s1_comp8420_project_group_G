"""Controlled output paths and a lightweight progress logger.

Generated artifacts are written only under ``outputs/`` so user uploads never sit
beside caches or reports. The logger records stderr progress lines and, when a
session is active, forwards each step to the structured session log. An in-memory
trace captures step names and elapsed milliseconds for demo reports.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass

OUTPUTS = "outputs"

_TRACE: list = []
_T0 = None


def reset_trace():
    """Clear the in-memory pipeline trace and restart the elapsed timer."""
    global _TRACE, _T0
    _TRACE = []
    _T0 = time.time()


def get_trace() -> list:
    """Return a copy of the current pipeline trace entries."""
    return list(_TRACE)


def ensure_outputs():
    """Create the outputs directory when it is missing."""
    os.makedirs(OUTPUTS, exist_ok=True)


def out_path(name: str) -> str:
    ensure_outputs()
    return os.path.join(OUTPUTS, name)


def write_json(name: str, obj) -> str:
    """Serialize a dataclass or mapping to ``outputs/<name>`` and return the path."""
    if is_dataclass(obj):
        obj = asdict(obj)
    elif hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    path = out_path(name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def write_text(name: str, text: str) -> str:
    """Write UTF-8 text to ``outputs/<name>`` and return the path."""
    path = out_path(name)
    with open(path, "w") as f:
        f.write(text)
    return path


def log(step: str, detail: str = ""):
    """Emit a stderr progress line and append to the active trace when enabled."""
    msg = f"[cli] {step}" + (f" :: {detail}" if detail else "")
    print(msg, file=sys.stderr, flush=True)
    if _T0 is not None:
        _TRACE.append({"step": step, "detail": detail,
                       "elapsed_ms": round((time.time() - _T0) * 1000, 1)})
    try:
        from .session_log import get_active
        active = get_active()
        if active is not None:
            active.log_pipeline_step(step, detail)
    except ImportError:
        pass
