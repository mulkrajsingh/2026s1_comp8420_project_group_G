"""Tests for the combined frontend and FastAPI launcher."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from app.web import ensure_frontend_build, frontend_needs_build, run_web


class WebLauncherTests(unittest.TestCase):
    def _frontend(self, root: Path) -> Path:
        frontend = root / "frontend"
        (frontend / "src").mkdir(parents=True)
        (frontend / "index.html").write_text("<main></main>", encoding="utf-8")
        (frontend / "package.json").write_text("{}", encoding="utf-8")
        (frontend / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'", encoding="utf-8")
        (frontend / "src" / "main.js").write_text("console.log('ui')", encoding="utf-8")
        return frontend

    def test_build_detection_handles_missing_current_and_stale_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frontend = self._frontend(Path(temp_dir))
            self.assertTrue(frontend_needs_build(frontend))

            built_index = frontend / "dist" / "index.html"
            built_index.parent.mkdir()
            built_index.write_text("built", encoding="utf-8")
            future = built_index.stat().st_mtime + 10
            os.utime(built_index, (future, future))
            self.assertFalse(frontend_needs_build(frontend))

            newer = future + 10
            source = frontend / "src" / "main.js"
            os.utime(source, (newer, newer))
            self.assertTrue(frontend_needs_build(frontend))
            self.assertTrue(frontend_needs_build(frontend, rebuild=True))

    def test_missing_pnpm_fails_with_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frontend = self._frontend(Path(temp_dir))
            with patch("app.web.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "pnpm is required"):
                    ensure_frontend_build(frontend)

    def test_failed_frontend_command_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frontend = self._frontend(Path(temp_dir))
            with (
                patch("app.web.shutil.which", return_value="/usr/local/bin/pnpm"),
                patch(
                    "app.web.subprocess.run",
                    side_effect=subprocess.CalledProcessError(2, ["pnpm", "install"]),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "exit code 2"):
                    ensure_frontend_build(frontend)

    def test_build_installs_dependencies_then_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frontend = self._frontend(Path(temp_dir))

            def fake_run(command, *, cwd, check):
                self.assertEqual(cwd, frontend)
                self.assertTrue(check)
                if command[-1] == "build":
                    (frontend / "dist").mkdir()
                    (frontend / "dist" / "index.html").write_text(
                        "built", encoding="utf-8"
                    )

            with (
                patch("app.web.shutil.which", return_value="/usr/local/bin/pnpm"),
                patch("app.web.subprocess.run", side_effect=fake_run) as run,
            ):
                self.assertTrue(ensure_frontend_build(frontend))

            self.assertEqual(
                run.call_args_list,
                [
                    call(
                        ["/usr/local/bin/pnpm", "install", "--frozen-lockfile"],
                        cwd=frontend,
                        check=True,
                    ),
                    call(
                        ["/usr/local/bin/pnpm", "build"],
                        cwd=frontend,
                        check=True,
                    ),
                ],
            )

    def test_server_launch_uses_requested_uvicorn_options(self) -> None:
        uvicorn_run = Mock()
        fake_uvicorn = SimpleNamespace(run=uvicorn_run)
        with (
            patch("app.web.ensure_frontend_build") as ensure,
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
        ):
            run_web(host="0.0.0.0", port=8123, rebuild=True, reload=True)

        ensure.assert_called_once_with(rebuild=True)
        uvicorn_run.assert_called_once_with(
            "app.api:api",
            host="0.0.0.0",
            port=8123,
            reload=True,
        )


if __name__ == "__main__":
    unittest.main()
