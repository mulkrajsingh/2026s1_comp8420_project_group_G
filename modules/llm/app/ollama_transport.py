"""HTTP transport for local Ollama generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def generate(
    host: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[str, str | None, dict[str, Any]]:
    """Send one generation request and normalize transport failures."""
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return "", f"{type(exc).__name__}: {exc}", {}

    text = str(body.get("response", "")).strip()
    error = None if text else "Ollama returned an empty response."
    return text, error, body
