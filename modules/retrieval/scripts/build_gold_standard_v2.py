"""Build an expanded gold standard via pooling + citation-graph judging (Step 3 of 3).

Inputs:
  - data/processed/pooled_candidates.json  (Step 1, build_pooled_candidates.py)
      per-query seed_ids + deduped candidates + which retriever(s) surfaced each
  - data/cache/s2_citations/{arxiv_id}.json (Step 2, fetch_citation_links.py)
      per-paper references/citations with externalIds.ArXiv

For each query, a pooled candidate that is not already one of the 5 original
seed papers is promoted to "citation_validated" if it has a *direct* citation
link to at least one seed paper for that query:
  - the candidate cites the seed (seed in candidate's references), or
  - the seed cites the candidate (candidate in seed's references), or
  - either direction confirmed via the S2 "citations" list.

No LLM judging is used — relevance is decided purely by retriever pooling
(unbiased candidate generation) + citation-graph links (objective relevance
signal grounded in the literature's own citation structure).

Output: data/processed/eval_queries_v2.json — list of per-query dicts:
  {
    "query": ...,
    "relevant_ids": [...],            # original 5 seeds, unchanged
    "citation_validated_ids": [...],  # newly promoted candidates
    "relevant_ids_v2": [...],         # union of the above, sorted
    "provenance": {
        paper_id: {"label": "seed" | "citation_validated",
                    "linked_seeds": [...], "pool_sources": [...]}
    }
  }

Usage:
    python modules/retrieval/scripts/build_gold_standard_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"

POOLED_PATH = RETRIEVAL_DIR / "data/processed/pooled_candidates.json"
CACHE_DIR = RETRIEVAL_DIR / "data/cache/s2_citations"
OUT_PATH = RETRIEVAL_DIR / "data/processed/eval_queries_v2.json"


def _arxiv_ids(s2_data: dict, field: str) -> set[str]:
    """Collect arXiv IDs from s2_data[field][*].externalIds.ArXiv."""
    out: set[str] = set()
    for entry in s2_data.get(field) or []:
        aid = (entry.get("externalIds") or {}).get("ArXiv")
        if aid:
            out.add(aid)
    return out


def load_link_sets(paper_id: str) -> tuple[set[str], set[str]]:
    """Return (references, citations) arXiv-ID sets for a pooled candidate."""
    cache = CACHE_DIR / (paper_id.replace("/", "_") + ".json")
    if not cache.exists():
        return set(), set()
    s2_data = json.loads(cache.read_text())
    if not s2_data:
        return set(), set()
    return _arxiv_ids(s2_data, "references"), _arxiv_ids(s2_data, "citations")


def main() -> None:
    if not POOLED_PATH.exists():
        raise SystemExit(f"Missing {POOLED_PATH} — run build_pooled_candidates.py first")

    pooled = json.loads(POOLED_PATH.read_text())

    # Pre-load reference/citation sets for every paper in the candidate universe
    universe = set()
    for entry in pooled.values():
        universe.update(entry["candidates"])
    links = {pid: load_link_sets(pid) for pid in universe}
    n_with_data = sum(1 for refs, cits in links.values() if refs or cits)
    print(f"Citation data available for {n_with_data}/{len(universe)} papers")

    results = []
    for query_text, entry in pooled.items():
        seed_ids = list(entry["seed_ids"])
        seed_set = set(seed_ids)
        sources = entry["sources"]

        provenance: dict[str, dict] = {}
        for pid in seed_ids:
            provenance[pid] = {"label": "seed", "linked_seeds": [], "pool_sources": sources.get(pid, ["seed"])}

        citation_validated = []
        for pid in entry["candidates"]:
            if pid in seed_set:
                continue
            cand_refs, cand_cites = links.get(pid, (set(), set()))
            linked_seeds = []
            for seed in seed_ids:
                seed_refs, seed_cites = links.get(seed, (set(), set()))
                if (
                    seed in cand_refs
                    or pid in seed_refs
                    or seed in cand_cites
                    or pid in seed_cites
                ):
                    linked_seeds.append(seed)
            if linked_seeds:
                citation_validated.append(pid)
                provenance[pid] = {
                    "label": "citation_validated",
                    "linked_seeds": linked_seeds,
                    "pool_sources": sources.get(pid, []),
                }

        relevant_v2 = sorted(seed_set | set(citation_validated))
        results.append({
            "query": query_text,
            "relevant_ids": seed_ids,
            "citation_validated_ids": citation_validated,
            "relevant_ids_v2": relevant_v2,
            "provenance": provenance,
        })
        print(f"{query_text!r}: {len(seed_ids)} seeds + {len(citation_validated)} citation-validated "
              f"= {len(relevant_v2)} relevant_ids_v2")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
