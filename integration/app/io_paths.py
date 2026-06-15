"""Controlled output paths + a tiny progress logger.

All generated artifacts go under `outputs/`; nothing is written next to uploaded
user documents. This is the seam Stage 03 (privacy / no-retention) builds on.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass

OUTPUTS = "outputs"

# Stage 06: in-memory trace of pipeline steps (step, detail, elapsed ms).
_TRACE: list = []
_T0 = None


def reset_trace():
    global _TRACE, _T0
    _TRACE = []
    _T0 = time.time()


def get_trace() -> list:
    return list(_TRACE)


def ensure_outputs():
    os.makedirs(OUTPUTS, exist_ok=True)


def out_path(name: str) -> str:
    ensure_outputs()
    return os.path.join(OUTPUTS, name)


def write_json(name: str, obj) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    elif hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    path = out_path(name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def write_text(name: str, text: str) -> str:
    path = out_path(name)
    with open(path, "w") as f:
        f.write(text)
    return path


def log(step: str, detail: str = ""):
    """Progress message for long-running local work (Stage 06 expands this)."""
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
