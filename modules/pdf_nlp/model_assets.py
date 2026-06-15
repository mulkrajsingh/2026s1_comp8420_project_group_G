"""Install and validate local model assets used by the PDF-NLP pipeline.

Large weights are deliberately excluded from Git. A user-provided archive is
extracted only after path traversal checks, then validated against the tracked
manifest before production NLP commands are allowed to run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = MODULE_ROOT / "models" / "runtime"
MANIFEST_PATH = MODULE_ROOT / "models" / "manifest.json"


class ModelAssetError(RuntimeError):
    """Raised when required local model assets are missing or invalid."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load the tracked runtime asset manifest."""
    if not path.is_file():
        raise ModelAssetError(f"Model manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_assets(
    *,
    root: Path = MODEL_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate required paths, sizes, and checksums from the manifest."""
    manifest = load_manifest(manifest_path)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for asset in manifest.get("assets", []):
        relative = Path(asset["path"])
        target = root / relative
        required_files = asset.get("required_files") or []
        asset_errors: list[str] = []
        if not target.exists():
            asset_errors.append(f"missing path: {target}")
        for required in required_files:
            file_path = target / required["path"]
            if not file_path.is_file():
                asset_errors.append(f"missing file: {file_path}")
                continue
            expected_size = required.get("bytes")
            if expected_size is not None and file_path.stat().st_size != expected_size:
                asset_errors.append(
                    f"size mismatch for {file_path}: "
                    f"{file_path.stat().st_size} != {expected_size}"
                )
            expected_hash = required.get("sha256")
            if expected_hash and sha256_file(file_path) != expected_hash:
                asset_errors.append(f"SHA-256 mismatch for {file_path}")
        errors.extend(asset_errors)
        checks.append(
            {
                "name": asset.get("name", relative.name),
                "path": str(target),
                "valid": not asset_errors,
                "errors": asset_errors,
            }
        )
    return {
        "valid": not errors,
        "root": str(root),
        "manifest": str(manifest_path),
        "checks": checks,
        "errors": errors,
    }


def repair_nested_runtime(root: Path = MODEL_ROOT) -> bool:
    """Promote assets from root/runtime/ after a naive ZIP extract."""
    nested = root / "runtime"
    if not nested.is_dir() or not validate_model_assets(root=nested)["valid"]:
        return False
    for child in nested.iterdir():
        destination = root / child.name
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        shutil.move(str(child), str(destination))
    nested.rmdir()
    return True


def require_model_assets(root: Path = MODEL_ROOT) -> dict[str, Any]:
    """Return validated asset metadata or raise a setup-focused error."""
    result = validate_model_assets(root=root)
    if not result["valid"]:
        joined = "\n- ".join(result["errors"])
        raise ModelAssetError(
            "Required PDF-NLP model assets are unavailable or invalid.\n"
            f"- {joined}\n"
            "Install the team model archive with:\n"
            "  python -m app.cli model-assets --archive /path/to/pdf_nlp_models.zip"
        )
    return result


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a ZIP while rejecting absolute and parent-traversal members."""
    destination = destination.resolve()
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ModelAssetError(f"Unsafe archive member: {member.filename}")
        resolved = (destination / member_path).resolve()
        if destination not in resolved.parents and resolved != destination:
            raise ModelAssetError(f"Archive member escapes destination: {member.filename}")
    archive.extractall(destination)


def install_model_archive(
    archive_path: Path,
    *,
    root: Path = MODEL_ROOT,
) -> dict[str, Any]:
    """Install a local team ZIP into the ignored runtime model directory."""
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise ModelAssetError(f"Model archive not found: {archive_path}")
    staging = root.parent / ".runtime-install"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, staging)
        candidate = staging / "runtime"
        source = candidate if candidate.is_dir() else staging
        validation = validate_model_assets(root=source)
        if not validation["valid"]:
            joined = "\n- ".join(validation["errors"])
            raise ModelAssetError(f"Archive validation failed:\n- {joined}")
        if root.exists():
            shutil.rmtree(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(root))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_model_assets(root=root)
