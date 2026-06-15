"""Fetch arXiv papers from the public API into PaperRecord JSONL.

Queries cs.CL, cs.AI, cs.LG, and stat.ML and writes roughly equal samples per
category. Intended for small bootstrap corpora when the full Kaggle dump is not
available locally.

Usage:
    python scripts/fetch_arxiv_corpus.py
    python scripts/fetch_arxiv_corpus.py --per-cat 250 --out data/processed/dev_1k.jsonl
"""

import argparse
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}

CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "stat.ML"]
BATCH_SIZE = 100   # arXiv API max per request
SLEEP_SEC  = 3.5   # be polite — arXiv asks for 3s between calls


def fetch_batch(category: str, start: int, max_results: int) -> list[dict]:
    """Fetch one batch of papers for a category. Returns list of PaperRecord dicts."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_bytes = resp.read()
    except Exception as exc:
        print(f"  Warning: request failed ({exc}), skipping batch.")
        return []

    root = ET.fromstring(xml_bytes)
    records = []

    for entry in root.findall("atom:entry", NS):
        try:
            arxiv_id_raw = entry.find("atom:id", NS).text.strip()
            # URL form: http://arxiv.org/abs/2005.11401v2 -> strip version
            arxiv_id = arxiv_id_raw.split("/abs/")[-1].split("v")[0]

            title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
            published = (entry.find("atom:published", NS).text or "")[:10]

            authors = [
                a.find("atom:name", NS).text.strip()
                for a in entry.findall("atom:author", NS)
                if a.find("atom:name", NS) is not None
            ]

            # Categories
            primary = entry.find("arxiv:primary_category", NS)
            cats = [primary.attrib.get("term", category)] if primary is not None else [category]
            for cat_el in entry.findall("atom:category", NS):
                term = cat_el.attrib.get("term", "")
                if term and term not in cats:
                    cats.append(term)

            # DOI
            doi_el = entry.find("arxiv:doi", NS)
            doi = doi_el.text.strip() if doi_el is not None else None

            # URL
            url_val = f"https://arxiv.org/abs/{arxiv_id}"

            record = {
                "paper_id":       arxiv_id,
                "title":          title,
                "abstract":       abstract,
                "authors":        authors,
                "categories":     cats[:5],
                "published_date": published,
                "venue":          None,
                "doi":            doi,
                "arxiv_id":       arxiv_id,
                "url":            url_val,
                "source":         "arxiv_api",
            }
            records.append(record)
        except Exception:
            continue

    return records


def fetch_category(category: str, target: int) -> list[dict]:
    """Fetch up to `target` papers for one category."""
    all_records = []
    seen_ids = set()
    start = 0

    print(f"\n  [{category}] Fetching up to {target} papers...")
    while len(all_records) < target:
        batch_size = min(BATCH_SIZE, target - len(all_records))
        print(f"    start={start}, requesting {batch_size}...", end=" ", flush=True)
        batch = fetch_batch(category, start, batch_size)

        new = [r for r in batch if r["paper_id"] not in seen_ids]
        for r in new:
            seen_ids.add(r["paper_id"])
        all_records.extend(new)

        print(f"got {len(new)} new (total {len(all_records)})")

        if len(batch) < batch_size:
            print(f"    API returned fewer results — stopping early.")
            break

        start += batch_size
        if len(all_records) < target:
            time.sleep(SLEEP_SEC)

    return all_records[:target]


def main() -> None:
    """Fetch papers from arXiv and write a PaperRecord JSONL corpus."""
    parser = argparse.ArgumentParser(description="Fetch arXiv corpus for COMP8420 project.")
    parser.add_argument("--per-cat", type=int, default=500,
                        help="Papers to fetch per category (default 500 = 2000 total).")
    parser.add_argument("--out", default="data/processed/dev_2k.jsonl",
                        help="Output JSONL path.")
    parser.add_argument("--cats", nargs="+", default=CATEGORIES,
                        help="arXiv categories to fetch.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_papers = []
    seen_global = set()

    for cat in args.cats:
        papers = fetch_category(cat, args.per_cat)
        for p in papers:
            if p["paper_id"] not in seen_global:
                seen_global.add(p["paper_id"])
                all_papers.append(p)
        print(f"  [{cat}] Added {len([p for p in papers if p['paper_id'] in seen_global])} unique. "
              f"Total so far: {len(all_papers)}")
        time.sleep(SLEEP_SEC)

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for record in all_papers:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_papers)} papers saved to {out_path}")

    # Quick stats
    from collections import Counter
    cat_counts = Counter(cat for p in all_papers for cat in p["categories"][:1])
    print("\nPrimary category breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat:15s}: {count}")

    years = Counter((p["published_date"] or "")[:4] for p in all_papers)
    print("\nYear breakdown (top 5):")
    for yr, cnt in sorted(years.items(), key=lambda x: -x[1])[:5]:
        print(f"  {yr}: {cnt}")


if __name__ == "__main__":
    main()
