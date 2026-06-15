"""Download Cornell arXiv metadata from Kaggle into ``data/raw``."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from lora_dataset.env import require_kaggle_credentials, sync_kaggle_access_token
from lora_dataset.paths import DATA_RAW, KAGGLE_RAW, WORKSTREAM_ROOT

KAGGLE_DATASET = "Cornell-University/arxiv"


def _kaggle_cmd() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def download_kaggle_raw(*, force: bool = False) -> str | None:
    """
    Ensure `data/raw/arxiv-metadata-oai-snapshot.json` exists.
    Returns SHA256 hex of the file, or None if already present and not re-hashed.
    """
    require_kaggle_credentials()
    sync_kaggle_access_token()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if KAGGLE_RAW.is_file() and not force:
        print(f"Kaggle raw snapshot already present: {KAGGLE_RAW}")
        return _sha256(KAGGLE_RAW)

    print(f"Downloading {KAGGLE_DATASET} to {DATA_RAW}...")
    subprocess.run(
        [*_kaggle_cmd(), "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(DATA_RAW), "--unzip"],
        check=True,
        cwd=WORKSTREAM_ROOT,
    )

    if not KAGGLE_RAW.is_file():
        raise SystemExit(f"ERROR: expected file missing after download: {KAGGLE_RAW}")

    digest = _sha256(KAGGLE_RAW)
    size = KAGGLE_RAW.stat().st_size
    print(f"Kaggle metadata ready: {KAGGLE_RAW} ({size} bytes, SHA256 {digest})")
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
