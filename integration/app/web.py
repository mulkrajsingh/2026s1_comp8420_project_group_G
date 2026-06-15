"""Build and serve the canonical Vite frontend with the local FastAPI app."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = INTEGRATION_ROOT / "frontend"


def _frontend_sources(frontend_dir: Path) -> list[Path]:
    sources = [
        frontend_dir / "index.html",
        frontend_dir / "package.json",
        frontend_dir / "pnpm-lock.yaml",
        frontend_dir / "pnpm-workspace.yaml",
        frontend_dir / "vite.config.js",
    ]
    src_dir = frontend_dir / "src"
    if src_dir.is_dir():
        sources.extend(path for path in src_dir.rglob("*") if path.is_file())
    return [path for path in sources if path.is_file()]


def frontend_needs_build(
    frontend_dir: Path = FRONTEND_DIR,
    *,
    rebuild: bool = False,
) -> bool:
    """Return whether the production frontend assets need to be rebuilt."""
    if rebuild:
        return True
    built_index = frontend_dir / "dist" / "index.html"
    if not built_index.is_file():
        return True
    built_at = built_index.stat().st_mtime
    return any(path.stat().st_mtime > built_at for path in _frontend_sources(frontend_dir))


def _run_pnpm(command: list[str], *, frontend_dir: Path) -> None:
    try:
        subprocess.run(command, cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError as exc:
        joined = " ".join(command)
        raise RuntimeError(
            f"Frontend command failed with exit code {exc.returncode}: {joined}"
        ) from exc


def ensure_frontend_build(
    frontend_dir: Path = FRONTEND_DIR,
    *,
    rebuild: bool = False,
) -> bool:
    """Build frontend assets when missing, stale, or explicitly requested."""
    if not frontend_needs_build(frontend_dir, rebuild=rebuild):
        return False

    pnpm = shutil.which("pnpm")
    if not pnpm:
        raise RuntimeError(
            "pnpm is required to build the web app. Install Node.js and pnpm, "
            "then rerun the command."
        )

    if not (frontend_dir / "node_modules").is_dir():
        _run_pnpm([pnpm, "install", "--frozen-lockfile"], frontend_dir=frontend_dir)
    _run_pnpm([pnpm, "build"], frontend_dir=frontend_dir)

    built_index = frontend_dir / "dist" / "index.html"
    if not built_index.is_file():
        raise RuntimeError(
            f"Frontend build completed without producing {built_index}"
        )
    return True


def run_web(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    rebuild: bool = False,
    reload: bool = False,
) -> None:
    """Ensure frontend assets exist, then serve the combined local application."""
    ensure_frontend_build(rebuild=rebuild)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'uvicorn'. From the repository root run: "
            "pip install -r requirements.txt"
        ) from exc

    uvicorn.run("app.api:api", host=host, port=port, reload=reload)
