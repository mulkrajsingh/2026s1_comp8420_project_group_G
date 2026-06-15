"""
Consolidate Yash's Semantic Scholar cache into a full dev_5k_enriched.jsonl.

Replicates the enrich_record() / _extract_reference_ids() logic from
modules/dataset/01_data_preprocessing_2.ipynb (Stage 02b), but reads S2 results
from the on-disk cache instead of calling the API. Resumable: records that are
already s2_enriched=True are passed through unchanged, and records with no cache
entry yet are left unenriched (s2_enriched=False) so a later re-run (once Yash's
cache reaches 5000/5000) picks them up automatically.

Reads (Yash's canonical module / data — not modified):
    modules/dataset/data/processed/dev_5k.jsonl   (5000 PaperRecords)
    modules/dataset/data/cache/s2/<arxiv_id>.json (local per-paper S2 cache)

Writes:
    modules/dataset/data/processed/dev_5k_enriched.jsonl

Usage:
    python modules/retrieval/scripts/build_enriched_corpus.py
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def cache_path(cache_dir: Path, arxiv_id: str) -> Path:
    return cache_dir / (arxiv_id.replace("/", "_") + ".json")


def extract_reference_ids(s2_data: dict) -> list[str]:
    """Return arXiv IDs from references; silently drops refs without one."""
    refs = []
    for ref in s2_data.get("references") or []:
        aid = (ref.get("externalIds") or {}).get("ArXiv")
        if aid:
            refs.append(aid)
    return refs


def enrich_record(record: dict, s2_data: dict) -> dict:
    """Merge S2 fields into a PaperRecord dict; returns the same dict."""
    record["citation_count"] = s2_data.get("citationCount", 0) or 0
    record["influential_citation_count"] = s2_data.get("influentialCitationCount", 0) or 0
    record["references"] = extract_reference_ids(s2_data)
    tldr_obj = s2_data.get("tldr")
    record["tldr"] = (tldr_obj.get("text", "") if tldr_obj else "") or ""
    record["s2_enriched"] = True
    return record


def build(input_jsonl: Path, cache_dir: Path, outputs: list[Path]) -> None:
    records = [json.loads(l) for l in input_jsonl.read_text().splitlines() if l.strip()]

    already = newly_enriched = cached_miss = no_cache = 0
    for rec in records:
        if rec.get("s2_enriched"):
            already += 1
            continue
        arxiv_id = rec.get("arxiv_id") or rec.get("paper_id")
        cpath = cache_path(cache_dir, arxiv_id)
        if not cpath.exists():
            no_cache += 1
            continue
        s2_data = json.loads(cpath.read_text())
        if not s2_data:
            cached_miss += 1
            continue
        enrich_record(rec, s2_data)
        newly_enriched += 1

    for out_path in outputs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records -> {out_path}")

    total_enriched = already + newly_enriched
    print(f"\nTotal records:        {len(records)}")
    print(f"  already enriched:   {already}")
    print(f"  newly enriched:     {newly_enriched}")
    print(f"  cached 404 (no S2): {cached_miss}")
    print(f"  no cache entry yet: {no_cache}")
    print(f"  s2_enriched=True:   {total_enriched} ({total_enriched / len(records):.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "modules/dataset/data/processed/dev_5k.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "modules/dataset/data/cache/s2"),
    )
    parser.add_argument(
        "--out",
        nargs="+",
        default=[
            str(REPO_ROOT / "modules/dataset/data/processed/dev_5k_enriched.jsonl"),
        ],
    )
    args = parser.parse_args()
    build(Path(args.input), Path(args.cache_dir), [Path(p) for p in args.out])


if __name__ == "__main__":
    main()
