"""Build filtered Kaggle arXiv PaperRecord JSONL for project arXiv RAG retrieval."""

from __future__ import annotations

import json
import random
import re
from calendar import monthrange
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterator

from app.schemas import validate_paper_record

from lora_dataset.io import write_jsonl
from lora_dataset.paths import (
    CORPUS_JSONL,
    CORPUS_LIMIT,
    CORPUS_MANIFEST,
    KAGGLE_RAW,
    SEED,
    WORKSTREAM_ROOT,
)

TARGET_CATEGORIES = {"cs.CL", "cs.AI", "cs.LG", "stat.ML"}
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")
TRUNCATED_CREATED_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*(\d{1,2})\s+([A-Za-z]{2,3})"
)
MONTHS_BY_PREFIX = {
    "ja": (1,),
    "fe": (2,),
    "ma": (3, 5),
    "ap": (4,),
    "ju": (6, 7),
    "au": (8,),
    "se": (9,),
    "oc": (10,),
    "no": (11,),
    "de": (12,),
}


def _primary_category(categories: list[str]) -> str | None:
    for category in categories:
        if category in TARGET_CATEGORIES:
            return category
    return None


def _normalize_authors(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(author).strip() for author in raw if str(author).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def _arxiv_id_from_record(record: dict[str, Any]) -> str | None:
    if record.get("id"):
        text = str(record["id"])
        match = ARXIV_ID_RE.search(text)
        if match:
            return match.group(1)
    versions = record.get("versions")
    if isinstance(versions, list) and versions:
        text = str(versions[0].get("created", "")) + str(record.get("id", ""))
    return None


def normalize_published_date(value: Any, arxiv_id: str | None = None) -> str | None:
    """Return an ISO date without inventing unavailable metadata."""
    text = str(value or "").strip()
    if not text or text == "n.d.":
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        pass

    match = TRUNCATED_CREATED_RE.match(text)
    arxiv_match = ARXIV_ID_RE.search(str(arxiv_id or ""))
    if not match or not arxiv_match:
        return None
    day = int(match.group(1))
    month_prefix = match.group(2).lower()[:2]
    year_month = arxiv_match.group(1).split(".", 1)[0]
    year = 2000 + int(year_month[:2])
    arxiv_month = int(year_month[2:])
    if not 1 <= arxiv_month <= 12:
        return None
    candidates = MONTHS_BY_PREFIX.get(month_prefix, ())
    month = min(candidates, key=lambda candidate: abs(candidate - arxiv_month)) if candidates else arxiv_month
    day = min(day, monthrange(year, month)[1])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _paper_record_from_arxiv(record: dict[str, Any], *, index: int) -> dict[str, Any] | None:
    categories_raw = record.get("categories") or ""
    if isinstance(categories_raw, str):
        categories = [part.strip() for part in categories_raw.split() if part.strip()]
    elif isinstance(categories_raw, list):
        categories = [str(part).strip() for part in categories_raw if str(part).strip()]
    else:
        categories = []

    primary = _primary_category(categories)
    if primary is None:
        return None

    title = str(record.get("title") or "").strip()
    abstract = str(record.get("abstract") or "").strip()
    if not title or not abstract:
        return None

    arxiv_id = _arxiv_id_from_record(record)
    paper_id = f"arxiv_{arxiv_id.replace('.', '_')}" if arxiv_id else f"kaggle_{index:08d}"
    published = None
    versions = record.get("versions")
    if isinstance(versions, list) and versions:
        published = normalize_published_date(versions[0].get("created"), arxiv_id)

    paper = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": _normalize_authors(record.get("authors")),
        "categories": categories,
        "published_date": published,
        "venue": str(record.get("journal-ref") or record.get("journal_ref") or "") or None,
        "doi": str(record.get("doi") or "") or None,
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "source": "kaggle_arxiv",
    }
    validate_paper_record(paper)
    return paper


def normalize_corpus_dates(
    *,
    input_path: Path = CORPUS_JSONL,
    output_path: Path = CORPUS_JSONL,
) -> int:
    """Normalize dates in an existing sampled corpus without resampling it."""
    if not input_path.is_file():
        raise SystemExit(f"Corpus not found: {input_path}")
    papers: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            paper = json.loads(line)
            paper["published_date"] = normalize_published_date(
                paper.get("published_date"),
                paper.get("arxiv_id"),
            )
            validate_paper_record(paper)
            papers.append(paper)
    write_jsonl(output_path, papers)
    if output_path == CORPUS_JSONL:
        CORPUS_MANIFEST.write_text(
            "\n".join(
                [
                    "# Project arXiv RAG Corpus Manifest",
                    "",
                    f"Output: `{output_path.relative_to(WORKSTREAM_ROOT)}`",
                    f"PaperRecord rows: {len(papers)}",
                    f"Target limit: {CORPUS_LIMIT}",
                    f"Seed: {SEED}",
                    f"Categories: {', '.join(sorted(TARGET_CATEGORIES))}",
                    "Source: cached Kaggle Cornell arXiv sample",
                    "Sampling: existing deterministic reservoir sample",
                    "Date normalization: ISO YYYY-MM-DD reconstructed from cached creation day and arXiv id.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return len(papers)


def _iter_arxiv_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream Kaggle arXiv metadata: JSON-per-line or single JSON array."""
    text_head = path.read_text(encoding="utf-8", errors="replace")[:2048].lstrip()
    if text_head.startswith("["):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def build_subset_from_records(
    records: Iterator[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    reservoir: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen = 0

    for index, raw in enumerate(records):
        paper = _paper_record_from_arxiv(raw, index=index)
        if paper is None:
            continue
        if not reservoir:
            selected.append(paper)
            if len(selected) >= limit:
                break
            continue
        seen += 1
        if len(selected) < limit:
            selected.append(paper)
        else:
            replace_index = rng.randint(0, seen - 1)
            if replace_index < limit:
                selected[replace_index] = paper

    return selected


def build_subset(
    input_path: Path,
    *,
    limit: int,
    seed: int,
    reservoir: bool = True,
) -> list[dict[str, Any]]:
    return build_subset_from_records(
        _iter_arxiv_records(input_path),
        limit=limit,
        seed=seed,
        reservoir=reservoir,
    )


def build_corpus(
    *,
    input_path: Path = KAGGLE_RAW,
    output_path: Path = CORPUS_JSONL,
    limit: int = CORPUS_LIMIT,
    seed: int = SEED,
) -> int:
    """Reservoir-sample `limit` PaperRecords from Kaggle arXiv metadata."""
    if not input_path.is_file():
        raise SystemExit(
            f"Input not found: {input_path}\n\n"
            "Run: python -m lora_dataset.create_dataset\n"
            "(requires KAGGLE_API_TOKEN in .env — see lora_dataset/README.md)"
        )

    papers = build_subset(input_path, limit=limit, seed=seed)
    if len(papers) < limit:
        print(f"Warning: only collected {len(papers)} papers (requested {limit}).")

    write_jsonl(output_path, papers)
    CORPUS_MANIFEST.write_text(
        "\n".join(
            [
                "# Project arXiv RAG Corpus Manifest",
                "",
                f"Output: `{output_path.relative_to(WORKSTREAM_ROOT)}`",
                f"PaperRecord rows: {len(papers)}",
                f"Target limit: {limit}",
                f"Seed: {seed}",
                f"Categories: {', '.join(sorted(TARGET_CATEGORIES))}",
                "Source: Kaggle Cornell arXiv `arxiv-metadata-oai-snapshot.json`",
                "Sampling: reservoir sampling (random 3k, not first 3k in file)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(papers)} PaperRecord rows to {output_path}")
    return len(papers)
