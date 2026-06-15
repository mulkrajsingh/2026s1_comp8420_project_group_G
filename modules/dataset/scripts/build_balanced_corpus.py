"""Build a deterministic category- and time-balanced arXiv PaperRecord corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


TARGET_CATEGORIES = ("cs.AI", "cs.CL", "cs.LG", "stat.ML")
TIME_BUCKETS = (
    ("foundational", None, 2017, 0.15),
    ("established", 2018, 2020, 0.20),
    ("recent", 2021, 2022, 0.25),
    ("current", 2023, None, 0.40),
)


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _published_date(raw: dict[str, Any]) -> str | None:
    versions = raw.get("versions")
    if isinstance(versions, list) and versions:
        created = versions[0].get("created")
        if created:
            try:
                return parsedate_to_datetime(str(created)).date().isoformat()
            except (TypeError, ValueError, OverflowError):
                pass
    update_date = str(raw.get("update_date") or "")
    try:
        return datetime.strptime(update_date[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _authors(raw: dict[str, Any]) -> list[str]:
    parsed = raw.get("authors_parsed")
    authors: list[str] = []
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, list) or not entry:
                continue
            last = _normalise_text(entry[0] if len(entry) > 0 else "")
            first = _normalise_text(entry[1] if len(entry) > 1 else "")
            suffix = _normalise_text(entry[2] if len(entry) > 2 else "")
            name = " ".join(part for part in (first, last, suffix) if part)
            if name:
                authors.append(name)
    if authors:
        return authors
    fallback = _normalise_text(raw.get("authors"))
    return [fallback] if fallback else []


def _primary_category(categories: Iterable[str]) -> str | None:
    category_set = set(categories)
    return next(
        (category for category in TARGET_CATEGORIES if category in category_set),
        None,
    )


def _time_bucket(year: int) -> str | None:
    for name, start, end, _ in TIME_BUCKETS:
        if (start is None or year >= start) and (end is None or year <= end):
            return name
    return None


def _to_paper_record(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], str, str] | None:
    paper_id = _normalise_text(raw.get("id"))
    title = _normalise_text(raw.get("title"))
    abstract = _normalise_text(raw.get("abstract"))
    categories = _normalise_text(raw.get("categories")).split()
    published_date = _published_date(raw)
    if not paper_id or not title or not abstract or not published_date:
        return None
    primary = _primary_category(categories)
    bucket = _time_bucket(int(published_date[:4]))
    if primary is None or bucket is None:
        return None
    record = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": _authors(raw),
        "categories": categories,
        "published_date": published_date,
        "venue": _normalise_text(raw.get("journal_ref")) or None,
        "doi": _normalise_text(raw.get("doi")) or None,
        "arxiv_id": paper_id,
        "url": f"https://arxiv.org/abs/{paper_id}",
        "source": "kaggle_arxiv",
    }
    return record, primary, bucket


def _weighted_quotas(total: int) -> dict[tuple[str, str], int]:
    if total <= 0:
        raise ValueError("total must be positive")
    category_totals = {
        category: total // len(TARGET_CATEGORIES)
        for category in TARGET_CATEGORIES
    }
    for category in TARGET_CATEGORIES[: total % len(TARGET_CATEGORIES)]:
        category_totals[category] += 1

    quotas: dict[tuple[str, str], int] = {}
    for category, category_total in category_totals.items():
        exact = [
            (name, category_total * weight)
            for name, _, _, weight in TIME_BUCKETS
        ]
        allocated = {name: int(value) for name, value in exact}
        remaining = category_total - sum(allocated.values())
        remainders = sorted(
            exact,
            key=lambda item: (item[1] - int(item[1]), item[0]),
            reverse=True,
        )
        for name, _ in remainders[:remaining]:
            allocated[name] += 1
        for name, _, _, _ in TIME_BUCKETS:
            quotas[(category, name)] = allocated[name]
    return quotas


def build_balanced_corpus(
    raw_path: Path,
    out_path: Path,
    *,
    total: int = 5000,
    seed: int = 8420,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Stream the raw snapshot and reservoir-sample each category/time bucket."""
    quotas = _weighted_quotas(total)
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[tuple[str, str]] = Counter()
    scanned = 0
    eligible = 0
    invalid_json = 0

    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            converted = _to_paper_record(raw)
            if converted is None:
                continue
            record, category, bucket = converted
            key = (category, bucket)
            quota = quotas[key]
            if quota <= 0:
                continue
            eligible += 1
            seen[key] += 1
            current = reservoirs[key]
            if len(current) < quota:
                current.append(record)
                continue
            replacement = rng.randrange(seen[key])
            if replacement < quota:
                current[replacement] = record

    deficits = {
        f"{category}:{bucket}": quotas[(category, bucket)] - len(reservoirs[(category, bucket)])
        for category in TARGET_CATEGORIES
        for bucket, _, _, _ in TIME_BUCKETS
        if len(reservoirs[(category, bucket)]) < quotas[(category, bucket)]
    }
    if deficits:
        raise RuntimeError(f"Raw snapshot cannot satisfy balanced quotas: {deficits}")

    records = [
        record
        for key in sorted(reservoirs)
        for record in reservoirs[key]
    ]
    records.sort(
        key=lambda record: (
            record["published_date"],
            record["paper_id"],
        ),
        reverse=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    report = {
        "input": str(raw_path),
        "output": str(out_path),
        "seed": seed,
        "target_total": total,
        "written": len(records),
        "records_scanned": scanned,
        "eligible_records": eligible,
        "invalid_json_records": invalid_json,
        "sha256": digest,
        "quotas": {
            f"{category}:{bucket}": quota
            for (category, bucket), quota in sorted(quotas.items())
        },
        "available_by_bucket": {
            f"{category}:{bucket}": seen[(category, bucket)]
            for category in TARGET_CATEGORIES
            for bucket, _, _, _ in TIME_BUCKETS
        },
        "year_min": min(int(record["published_date"][:4]) for record in records),
        "year_max": max(int(record["published_date"][:4]) for record in records),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8420)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_balanced_corpus(
        args.raw,
        args.out,
        total=args.total,
        seed=args.seed,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
