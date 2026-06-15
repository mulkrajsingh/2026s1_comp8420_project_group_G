#!/usr/bin/env python3
"""Generate ignored test artifacts from real PDFs and retrieval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.harness.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    PAPER_PDFS,
    corpus_path,
    parsed_artifact_path,
    rag_pack_artifact_path,
)
from tests.harness.runners import run_parse_pdf, run_recommend_topic  # noqa: E402


def _load_parsed_title(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    title = (meta.get("title") or "").strip()
    if title:
        return title
    abstract = (meta.get("abstract") or "").strip()
    return abstract[:500] if abstract else "related academic papers"


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("drq_v2", "siga"):
        pdf = PAPER_PDFS[name]
        out = parsed_artifact_path(name)
        print(f"Parsing {name}: {pdf}")
        proc = run_parse_pdf(pdf, out)
        if proc.returncode != 0:
            print(proc.stderr or proc.stdout, file=sys.stderr)
            return proc.returncode
        print(f"  -> {out}")

    drq_parsed = parsed_artifact_path("drq_v2")
    query = _load_parsed_title(drq_parsed)
    rec_out = ARTIFACTS_DIR / "drq_v2_recommendations.json"
    print(f"Retrieval for title query: {query!r}")
    proc = run_recommend_topic(
        query,
        corpus_path(sample=True),
        rec_out,
        top_k=10,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        return proc.returncode

    rag_src = rec_out.parent / "rag_evidence_pack.json"
    rag_dest = rag_pack_artifact_path("drq_v2")
    if rag_src.is_file():
        rag_dest.write_text(rag_src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  -> {rag_dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
