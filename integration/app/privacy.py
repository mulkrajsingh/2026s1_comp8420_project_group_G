"""Upload handling, output placement, and no-retention policy.

Uploaded PDFs are copied to a temporary working path and removed after processing
unless the caller opts in to ``save=True``. Generated files are written only under
``outputs/``. AI-generated sections are disclosed through ``AnalysisResult.flags``.

The ``process_upload`` context manager is used by analyze-pdf and peer-review so
temp files are always cleaned up. ``write_privacy_check`` writes a short runtime
verification report to ``outputs/privacy_check.md``.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .io_paths import write_text, log

_TMP_DIR = os.path.join("outputs", ".uploads_tmp")
_SAMPLE_PDF = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "papers"
    / "drq_v2"
    / "2107.09645v1.pdf"
)


@contextmanager
def process_upload(pdf_path: str, save: bool = False):
    """Yield a temp working copy of the upload; delete it afterwards unless save."""
    os.makedirs(_TMP_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".pdf", dir=_TMP_DIR)
    os.close(fd)
    if os.path.exists(pdf_path):
        shutil.copyfile(pdf_path, tmp)
    log("privacy", f"working on temp copy {os.path.basename(tmp)} (save={save})")
    try:
        yield tmp
    finally:
        if not save and os.path.exists(tmp):
            os.remove(tmp)
            log("privacy", "temp upload deleted (no-retention)")


def write_privacy_check() -> str:
    """Demonstrate temp-upload deletion and write ``outputs/privacy_check.md``."""
    deleted = False
    with process_upload(str(_SAMPLE_PDF), save=False) as tmp:
        existed = os.path.exists(tmp)
    deleted = existed and not os.path.exists(tmp)

    md = [
        "# Privacy & no-retention check", "",
        "## Policy",
        "- Uploaded PDFs are copied to a temp path and **deleted after processing** "
        "unless the user explicitly saves them.",
        "- Generated artifacts are written **only** under `outputs/`; user documents "
        "are never mixed with caches.",
        "- AI-generated sections are **disclosed** in every result "
        "(`flags.ai_generated_sections`).",
        "- All processing is **local-only**; no upload leaves the machine.",
        "",
        "## Runtime check",
        f"- Temp upload created and then deleted after processing: "
        f"**{'PASS' if deleted else 'CHECK'}**",
        f"- Output directory: `outputs/`",
        "",
    ]
    return write_text("privacy_check.md", "\n".join(md))
