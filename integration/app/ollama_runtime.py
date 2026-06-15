"""Ollama model residency for the combined web application.

Keeps one local generation model loaded during a web session, switches between
the project base and LoRA adapter tags per request, and unloads on shutdown.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_ADAPTER_MODEL = "qwen3-research-lora:latest"
DEFAULT_WEB_KEEP_ALIVE = "30m"
DEFAULT_WARM_CONTEXT_TOKENS = 8192
DEFAULT_UNLOAD_TIMEOUT_SECONDS = 30.0
PROJECT_MODELS = (
    {
        "id": DEFAULT_OLLAMA_MODEL,
        "label": "Qwen3 8B (base)",
        "kind": "base",
        "description": "Local Qwen3 8B foundation model.",
    },
    {
        "id": DEFAULT_OLLAMA_ADAPTER_MODEL,
        "label": "Qwen3 8B + team LoRA",
        "kind": "adapter",
        "description": "Qwen3 8B with the team-trained research LoRA adapter.",
    },
)
PROJECT_MODEL_IDS = frozenset(model["id"] for model in PROJECT_MODELS)


def _ollama_json(
    path: str,
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 10.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(f"Unable to query Ollama at {host}: {exc}") from exc


def list_ollama_models(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Return installed Ollama model metadata."""
    body = _ollama_json("/api/tags", host=host, timeout=timeout)
    models = body.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/tags returned an invalid model list")
    return [model for model in models if isinstance(model, dict)]


def list_running_ollama_models(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Return models currently resident in Ollama memory."""
    body = _ollama_json("/api/ps", host=host, timeout=timeout)
    models = body.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/ps returned an invalid model list")
    return [model for model in models if isinstance(model, dict)]


def _model_name(model: dict[str, Any]) -> str:
    return str(model.get("name") or model.get("model") or "")


def project_model_catalog(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Describe the supported base and adapter tags and their availability."""
    installed = {
        _model_name(model): model
        for model in list_ollama_models(host=host, timeout=timeout)
    }
    catalog = []
    for configured in PROJECT_MODELS:
        metadata = installed.get(configured["id"], {})
        catalog.append(
            {
                **configured,
                "available": bool(metadata),
                "size": metadata.get("size"),
                "modified_at": metadata.get("modified_at"),
            }
        )
    return catalog


def wait_for_model_unload(
    model: str,
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = DEFAULT_UNLOAD_TIMEOUT_SECONDS,
    poll_interval: float = 0.1,
) -> None:
    """Wait until Ollama confirms that a model runner left memory."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        running = {
            _model_name(item)
            for item in list_running_ollama_models(host=host)
        }
        if model not in running:
            return
        time.sleep(poll_interval)
    raise RuntimeError(
        f"Ollama did not unload {model!r} within {timeout:.1f} seconds"
    )


def set_model_residency(
    *,
    model: str,
    keep_alive: str | int,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Warm or unload an Ollama model and control its residency."""
    normalized_keep_alive: str | int = keep_alive
    if str(keep_alive) in {"-1", "0"}:
        normalized_keep_alive = int(keep_alive)
    payload = {
        "model": model,
        "keep_alive": normalized_keep_alive,
        "stream": False,
    }
    if normalized_keep_alive != 0:
        payload.update(
            {
                "prompt": "Ready",
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1,
                    "num_ctx": int(
                        os.getenv(
                            "COMP8420_OLLAMA_WARM_NUM_CTX",
                            DEFAULT_WARM_CONTEXT_TOKENS,
                        )
                    ),
                },
            }
        )
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"Unable to set Ollama residency for {model!r} at {host}: {exc}"
        ) from exc
    if body.get("error"):
        raise RuntimeError(f"Ollama could not load {model!r}: {body['error']}")
    return {
        "status": "loaded" if normalized_keep_alive != 0 else "unloaded",
        "model": model,
        "host": host.rstrip("/"),
        "keep_alive": normalized_keep_alive,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "done_reason": body.get("done_reason"),
    }


class OllamaModelManager:
    """Serialize local model use and keep only one project model resident."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        keep_alive: str | int = DEFAULT_WEB_KEEP_ALIVE,
    ) -> None:
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self._active_model: str | None = None
        self._state: dict[str, Any] = {"status": "not_started"}
        self._lock = threading.RLock()

    def configure(self, *, host: str, keep_alive: str | int) -> None:
        with self._lock:
            self.host = host.rstrip("/")
            self.keep_alive = keep_alive

    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def catalog(self) -> dict[str, Any]:
        with self._lock:
            return {
                "default_model": DEFAULT_OLLAMA_MODEL,
                "active_model": self._active_model,
                "models": project_model_catalog(host=self.host),
            }

    def _validate_available(self, model: str) -> None:
        if model not in PROJECT_MODEL_IDS:
            allowed = ", ".join(sorted(PROJECT_MODEL_IDS))
            raise ValueError(
                f"Unsupported Ollama model {model!r}. Select one of: {allowed}"
            )
        available = {
            item["id"]: item["available"]
            for item in project_model_catalog(host=self.host)
        }
        if not available.get(model):
            raise RuntimeError(
                f"Required Ollama model {model!r} is not installed"
            )

    def _switch_locked(self, model: str) -> dict[str, Any]:
        if self._active_model == model:
            return {**self._state, "switched": False}

        self._validate_available(model)
        previous_model = self._active_model
        running = {
            _model_name(item)
            for item in list_running_ollama_models(host=self.host)
        }
        unloaded = []
        for running_model in sorted(running & PROJECT_MODEL_IDS):
            if running_model == model:
                continue
            set_model_residency(
                model=running_model,
                keep_alive=0,
                host=self.host,
            )
            wait_for_model_unload(running_model, host=self.host)
            unloaded.append(running_model)

        try:
            loaded = set_model_residency(
                model=model,
                keep_alive=self.keep_alive,
                host=self.host,
            )
        except Exception:
            self._active_model = None
            self._state = {
                "status": "load_failed",
                "model": model,
                "previous_model": previous_model,
                "unloaded_models": unloaded,
                "host": self.host,
            }
            raise

        self._active_model = model
        self._state = {
            **loaded,
            "previous_model": previous_model,
            "unloaded_models": unloaded,
            "switched": previous_model != model,
        }
        return dict(self._state)

    def start(self, model: str = DEFAULT_OLLAMA_MODEL) -> dict[str, Any]:
        with self._lock:
            return self._switch_locked(model)

    @contextmanager
    def use(self, *, backend: str, model: str):
        """Hold the model lock for the full request to prevent mid-run unloads."""
        if backend != "ollama":
            yield {"status": "not_required", "model": model}
            return
        with self._lock:
            state = self._switch_locked(model)
            yield state

    def shutdown(self) -> dict[str, Any]:
        with self._lock:
            model = self._active_model
            if model is None:
                self._state = {"status": "not_loaded"}
                return dict(self._state)
            try:
                state = set_model_residency(
                    model=model,
                    keep_alive=0,
                    host=self.host,
                )
                wait_for_model_unload(model, host=self.host)
            except Exception:
                self._state = {
                    "status": "unload_failed",
                    "model": model,
                    "host": self.host,
                }
                raise
            finally:
                self._active_model = None
            self._state = state
            return dict(state)


MODEL_MANAGER = OllamaModelManager()
