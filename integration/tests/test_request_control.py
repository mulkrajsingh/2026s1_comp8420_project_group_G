"""Tests for cooperative web request cancellation."""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from integration.app.request_control import (
    RequestCancelledError,
    bind_request,
    cancel,
    is_cancelled,
    register,
    unregister,
)


class RequestControlTests(unittest.TestCase):
    def test_cancel_marks_request_and_terminates_process(self) -> None:
        request_id = "test-cancel"
        register(request_id)
        bind_request(request_id)
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            from integration.app.request_control import set_process

            set_process(proc)
            self.assertTrue(cancel(request_id))
            self.assertTrue(is_cancelled())
            deadline = time.monotonic() + 5
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(proc.poll())
        finally:
            unregister(request_id)

    def test_cancel_unknown_request_returns_false(self) -> None:
        self.assertFalse(cancel("missing-request"))

    def test_run_cli_raises_when_cancelled(self) -> None:
        from integration.app.providers.live_providers import _run_cli

        request_id = "test-run-cli"
        register(request_id)
        bind_request(request_id)
        try:
            with patch(
                "integration.app.request_control.is_cancelled",
                side_effect=[False, True],
            ):
                with self.assertRaises(RequestCancelledError):
                    _run_cli(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        ".",
                        "test command",
                        [],
                    )
        finally:
            unregister(request_id)


if __name__ == "__main__":
    unittest.main()
