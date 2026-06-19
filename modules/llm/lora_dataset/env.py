"""Load ``.env`` files and validate Kaggle API credentials for corpus download."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .paths import WORKSTREAM_ROOT

ACCESS_TOKEN_FILE = Path.home() / ".kaggle" / "access_token"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
REPO_ROOT = WORKSTREAM_ROOT.parent


def load_dotenv() -> Path | None:
    """Load the first .env found in workstream root or repo root."""
    for candidate in (WORKSTREAM_ROOT / ".env", REPO_ROOT / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value
        return candidate
    return None


def has_kaggle_credentials() -> bool:
    """Return whether Kaggle credentials are available in env or standard paths."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if ACCESS_TOKEN_FILE.is_file():
        return True
    if KAGGLE_JSON.is_file():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def sync_kaggle_access_token() -> None:
    """Write KAGGLE_API_TOKEN to ~/.kaggle/access_token for the Kaggle CLI."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        return
    ACCESS_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCESS_TOKEN_FILE.write_text(token, encoding="utf-8")
    ACCESS_TOKEN_FILE.chmod(0o600)


def require_kaggle_credentials() -> None:
    """Exit with setup instructions when Kaggle credentials are missing."""
    load_dotenv()
    if has_kaggle_credentials():
        sync_kaggle_access_token()
        return
    print(
        "ERROR: Kaggle API credentials not configured.\n\n"
        "1. Create a token at https://www.kaggle.com/settings (API → Generate New Token)\n"
        "2. Create .env at the repository root using .env.example as the template.\n"
        "3. Add: KAGGLE_API_TOKEN=<your token>\n"
        "4. Re-run: python -m modules.llm.lora_dataset.create_dataset\n",
        file=sys.stderr,
    )
    sys.exit(1)
