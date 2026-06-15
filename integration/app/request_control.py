"""Track in-flight web requests and cooperative cancellation."""

from __future__ import annotations

import contextvars
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class RequestCancelledError(RuntimeError):
    """Raised when a client cancels an active integration request."""


@dataclass
class _ActiveRequest:
    cancelled: bool = False
    process: subprocess.Popen[Any] | None = field(default=None, repr=False)


_current_request_id = contextvars.ContextVar("request_id", default=None)
_lock = threading.Lock()
_active: dict[str, _ActiveRequest] = {}


def new_request_id() -> str:
    return uuid4().hex


def register(request_id: str) -> None:
    with _lock:
        _active[request_id] = _ActiveRequest()


def unregister(request_id: str) -> None:
    with _lock:
        _active.pop(request_id, None)
    if _current_request_id.get() == request_id:
        _current_request_id.set(None)


def bind_request(request_id: str):
    """Bind the current execution context to one active request id."""
    _current_request_id.set(request_id)


def is_cancelled() -> bool:
    request_id = _current_request_id.get()
    if not request_id:
        return False
    with _lock:
        entry = _active.get(request_id)
        return bool(entry and entry.cancelled)


def set_process(process: subprocess.Popen[Any] | None) -> None:
    request_id = _current_request_id.get()
    if not request_id:
        return
    with _lock:
        entry = _active.get(request_id)
        if entry is not None:
            entry.process = process


def cancel(request_id: str) -> bool:
    """Cancel one active request and terminate its subprocess if present."""
    with _lock:
        entry = _active.get(request_id)
        if entry is None:
            return False
        entry.cancelled = True
        process = entry.process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return True
