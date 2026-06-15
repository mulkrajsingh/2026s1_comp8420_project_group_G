"""Tests for Ollama model residency without generation."""

from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from app.ollama_runtime import OllamaModelManager, set_model_residency


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OllamaRuntimeTests(unittest.TestCase):
    def test_warm_request_initializes_inference_and_keeps_model_resident(self) -> None:
        with patch(
            "app.ollama_runtime.urllib.request.urlopen",
            return_value=_Response({"done": True, "done_reason": "load"}),
        ) as urlopen:
            result = set_model_residency(
                model="qwen3:8b",
                keep_alive="-1",
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload,
            {
                "model": "qwen3:8b",
                "keep_alive": -1,
                "stream": False,
                "prompt": "Ready",
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1,
                    "num_ctx": 8192,
                },
            },
        )
        self.assertEqual(result["status"], "loaded")

    def test_ollama_error_fails_startup(self) -> None:
        with patch(
            "app.ollama_runtime.urllib.request.urlopen",
            return_value=_Response({"error": "model not found"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "model not found"):
                set_model_residency(model="missing", keep_alive="-1")

    def test_model_change_unloads_previous_before_warming_selected(self) -> None:
        manager = OllamaModelManager(
            host="http://ollama.test",
            keep_alive="30m",
        )
        manager._active_model = "qwen3:8b"
        manager._state = {"status": "loaded", "model": "qwen3:8b"}
        residency = Mock(
            side_effect=[
                {"status": "unloaded", "model": "qwen3:8b"},
                {"status": "loaded", "model": "qwen3-research-lora:latest"},
            ]
        )

        with (
            patch(
                "app.ollama_runtime.project_model_catalog",
                return_value=[
                    {"id": "qwen3:8b", "available": True},
                    {"id": "qwen3-research-lora:latest", "available": True},
                ],
            ),
            patch(
                "app.ollama_runtime.list_running_ollama_models",
                side_effect=[
                    [{"name": "qwen3:8b"}],
                    [],
                ],
            ),
            patch("app.ollama_runtime.set_model_residency", residency),
        ):
            with manager.use(
                backend="ollama",
                model="qwen3-research-lora:latest",
            ) as state:
                self.assertEqual(
                    state["model"],
                    "qwen3-research-lora:latest",
                )

        self.assertEqual(
            residency.call_args_list[0].kwargs,
            {
                "model": "qwen3:8b",
                "keep_alive": 0,
                "host": "http://ollama.test",
            },
        )
        self.assertEqual(
            residency.call_args_list[1].kwargs,
            {
                "model": "qwen3-research-lora:latest",
                "keep_alive": "30m",
                "host": "http://ollama.test",
            },
        )

    def test_same_model_is_reused_without_reload(self) -> None:
        manager = OllamaModelManager()
        manager._active_model = "qwen3:8b"
        manager._state = {"status": "loaded", "model": "qwen3:8b"}

        with patch("app.ollama_runtime.set_model_residency") as residency:
            with manager.use(backend="ollama", model="qwen3:8b") as state:
                self.assertFalse(state["switched"])

        residency.assert_not_called()

    def test_unsupported_model_is_rejected(self) -> None:
        manager = OllamaModelManager()
        with self.assertRaisesRegex(ValueError, "Unsupported Ollama model"):
            manager.start("qwen-finance-lora:latest")


class WebLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_lifespan_loads_then_unloads_main_model(self) -> None:
        from app.api import api, lifespan

        calls: list[dict] = []

        class FakeManager:
            def configure(self, **kwargs):
                calls.append({"operation": "configure", **kwargs})

            def start(self, model):
                calls.append({"operation": "start", "model": model})
                return {"status": "loaded", "model": model}

            def shutdown(self):
                calls.append({"operation": "shutdown"})
                return {"status": "unloaded", "model": "qwen3:8b"}

            def state(self):
                return {"status": "loaded", "model": "qwen3:8b"}

            @contextmanager
            def use(self, *, backend, model):
                yield {"status": "loaded", "model": model}

        with (
            patch.dict(
                os.environ,
                {
                    "COMP8420_OLLAMA_HOST": "http://ollama.test",
                    "COMP8420_OLLAMA_MODEL": "qwen3:8b",
                    "COMP8420_OLLAMA_KEEP_ALIVE": "-1",
                },
                clear=False,
            ),
            patch("app.api.MODEL_MANAGER", new=FakeManager()),
            patch(
                "app.api._warm_query_analyzer",
                return_value={
                    "status": "loaded",
                    "embedding_model": "test-minilm",
                    "fallback_model": "test-tinybert",
                },
            ) as warm_query_analyzer,
        ):
            async with lifespan(api):
                self.assertEqual(api.state.ollama["status"], "loaded")
                self.assertEqual(api.state.query_analyzer["status"], "loaded")

        warm_query_analyzer.assert_called_once_with()
        self.assertEqual(
            calls,
            [
                {
                    "operation": "configure",
                    "keep_alive": "-1",
                    "host": "http://ollama.test",
                },
                {
                    "operation": "start",
                    "model": "qwen3:8b",
                },
                {"operation": "shutdown"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
