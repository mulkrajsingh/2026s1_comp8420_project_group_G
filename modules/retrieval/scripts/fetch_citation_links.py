"""Fetch Semantic Scholar citation links for pooled gold candidates.

Reads pooled candidate IDs, queries the S2 batch API for reference and citation
arXiv IDs, and caches responses for citation-graph gold expansion.

Usage:
    python modules/retrieval/scripts/fetch_citation_links.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"

POOLED_PATH = RETRIEVAL_DIR / "data/processed/pooled_candidates.json"
CACHE_DIR = RETRIEVAL_DIR / "data/cache/s2_citations"
ENV_PATH = REPO_ROOT / "modules/dataset/.env"

S2_FIELDS = "externalIds,references.externalIds,citations.externalIds"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH_SIZE = 100


def load_s2_api_key() -> str:
    """Read S2_API_KEY from the dataset module without printing its value."""
    if not ENV_PATH.exists():
        return ""
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "S2_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def cache_path(arxiv_id: str) -> Path:
    return CACHE_DIR / (arxiv_id.replace("/", "_") + ".json")


def fetch_batch(arxiv_ids: list[str], headers: dict, retries: int = 6) -> list[dict | None]:
    """POST a batch of arXiv IDs; returns one result (or None) per input ID, in order."""
    body = {"ids": [f"ARXIV:{aid}" for aid in arxiv_ids]}
    params = {"fields": S2_FIELDS}
    for attempt in range(retries):
        try:
            resp = requests.post(S2_BATCH_URL, headers=headers, params=params, json=body, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = min(3 * (attempt + 1), 15)
                print(f"  rate-limited (batch); sleeping {wait}s...")
                time.sleep(wait)
                continue
            print(f"  unexpected status {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            print(f"  network error ({attempt + 1}/{retries}): {exc}")
            time.sleep(3)
    raise RuntimeError(f"Batch fetch failed after {retries} attempts for {len(arxiv_ids)} ids")


def main() -> None:
    """Fetch and cache S2 citation links for all pooled candidates."""
    if not POOLED_PATH.exists():
        sys.exit(f"Missing {POOLED_PATH} — run build_pooled_candidates.py first")

    pooled = json.loads(POOLED_PATH.read_text())

    universe: set[str] = set()
    for entry in pooled.values():
        universe.update(entry["candidates"])
    universe_sorted = sorted(universe)
    print(f"{len(universe_sorted)} unique candidate paper_ids across {len(pooled)} queries")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_fetch = [aid for aid in universe_sorted if not cache_path(aid).exists()]
    print(f"{len(universe_sorted) - len(to_fetch)} already cached, {len(to_fetch)} to fetch")

    api_key = load_s2_api_key()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
        print(f"S2 mode: authenticated, key length={len(api_key)}")
    else:
        print("S2 mode: unauthenticated — S2_API_KEY not found")

    n_found, n_missing = 0, 0
    for i in range(0, len(to_fetch), BATCH_SIZE):
        chunk = to_fetch[i:i + BATCH_SIZE]
        results = fetch_batch(chunk, headers)
        for aid, data in zip(chunk, results):
            if data is None:
                cache_path(aid).write_text("{}", encoding="utf-8")
                n_missing += 1
            else:
                cache_path(aid).write_text(json.dumps(data), encoding="utf-8")
                n_found += 1
        print(f"  batch {i // BATCH_SIZE + 1}: {len(chunk)} ids -> found={n_found} missing={n_missing} (cumulative)")
        if i + BATCH_SIZE < len(to_fetch):
            time.sleep(2)

    n_cached_now = sum(1 for aid in universe_sorted if cache_path(aid).exists())
    print(f"\nDone. {n_cached_now}/{len(universe_sorted)} papers cached in {CACHE_DIR}")


if __name__ == "__main__":
    main()
