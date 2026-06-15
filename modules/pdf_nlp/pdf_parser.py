"""PDF text extraction and canonical ``ParsedPaper`` assembly.

Extracts page text, splits sections and references, and fills metadata fields
from the first page. POS, NER, keyphrases, summaries, and structural checks
are handled separately in ``paper_analysis.py``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import regex
from pypdf import PdfReader
from pypdf.errors import PdfReadError

CANONICAL_SECTIONS = ("abstract", "introduction", "method", "results", "conclusion")
OPTIONAL_SECTIONS = ("related_work", "limitations", "discussion")
EMPTY_ENTITIES = {
    "methods": [],
    "datasets": [],
    "tasks": [],
    "metrics": [],
    "institutions": [],
}

_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "introduction": ("introduction", "background", "preliminaries"),
    "method": (
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "algorithm",
        "framework",
        "architecture",
        "proposed",
    ),
    "results": (
        "results",
        "experiments",
        "experimental",
        "evaluation",
        "empirical",
        "findings",
    ),
    "conclusion": (
        "conclusion",
        "conclusions",
        "future work",
        "summary",
    ),
    "related_work": ("related work", "prior work", "literature review"),
    "limitations": ("limitation", "limitations"),
    "discussion": ("discussion",),
}

_REFERENCE_HEADINGS = ("references", "bibliography", "works cited")
_REFERENCE_YEAR_RE = regex.compile(r"(?<!\d)(?:18|19|20)\d{2}[a-z]?(?!\d)", regex.IGNORECASE)


class PdfParserError(ValueError):
    """Raised when a PDF cannot be parsed."""


def stable_paper_id(pdf_path: Path) -> str:
    """Derive a stable identifier from the PDF filename."""
    stem = pdf_path.stem.strip()
    if not stem:
        stem = "uploaded"
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:10]
    safe = re.sub(r"[^\w\-.]+", "_", stem).strip("_").lower()
    return f"{safe}_{digest}" if safe else f"uploaded_pdf_{digest}"


def _arxiv_id_from_name(name: str) -> str | None:
    match = regex.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", name, regex.IGNORECASE)
    return match.group(1) if match else None


def validate_pdf_path(pdf_path: Path) -> None:
    """Fail fast on missing, empty, or unreadable PDF inputs."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.stat().st_size == 0:
        raise PdfParserError(f"PDF is empty: {pdf_path}")
    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as exc:
        raise PdfParserError(f"Invalid or corrupted PDF: {pdf_path}") from exc
    except Exception as exc:
        raise PdfParserError(f"Cannot open PDF: {pdf_path} ({exc})") from exc
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise PdfParserError(f"PDF is encrypted and cannot be read: {pdf_path}")
        except Exception as exc:
            raise PdfParserError(f"PDF is encrypted and cannot be read: {pdf_path}") from exc
    if len(reader.pages) == 0:
        raise PdfParserError(f"PDF has no pages: {pdf_path}")


def extract_page_texts(pdf_path: Path) -> tuple[list[str], list[str]]:
    """Return per-page text and any extraction warnings."""
    validate_pdf_path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    warnings: list[str] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
        if not text.strip():
            warnings.append(f"page {index + 1}: no extractable text")
    if not any(page.strip() for page in pages):
        raise PdfParserError(f"PDF contains no extractable text: {pdf_path}")
    return pages, warnings


def join_page_text(page_texts: list[str]) -> str:
    """Concatenate per-page strings with newline separators."""
    return "\n".join(page_texts)


def _normalize_heading(heading: str) -> str:
    heading = regex.sub(r"^\d+(?:\.\d+)*\s*", "", heading.strip())
    heading = regex.sub(r"\s+", " ", heading)
    return heading.strip(" .:-").lower()


def _canonical_section_key(heading: str) -> str | None:
    normalized = _normalize_heading(heading)
    if not normalized:
        return None
    if normalized in {"abstract", "acknowledgements", "acknowledgments"}:
        return "abstract" if normalized == "abstract" else None
    for key, aliases in _SECTION_ALIASES.items():
        if normalized in aliases:
            return key
        if any(normalized.startswith(alias) for alias in aliases):
            return key
    return None


def find_numbered_headings(full_text: str) -> list[tuple[str, int]]:
    """Return (heading, start_offset) pairs for numbered section headings."""
    headings: list[tuple[str, int]] = []
    pattern = regex.compile(r"(?m)^(\d+(?:\.\d+)*\s+[A-Z][^\n]{0,120})$")
    for match in pattern.finditer(full_text):
        heading = match.group(1).strip()
        # Skip figure/table captions that look like numbered headings.
        if regex.search(r"\b(figure|table|algorithm|appendix)\b", heading, regex.IGNORECASE):
            if not regex.match(r"^\d+\s+[A-Z]", heading):
                continue
        headings.append((heading, match.start()))
    return headings


def extract_abstract(full_text: str) -> str:
    match = regex.search(r"(?is)\babstract\b\s*(.+?)(?=\n\s*\d+\s+[A-Z]|\Z)", full_text)
    if not match:
        return ""
    abstract = regex.sub(r"\s+", " ", match.group(1)).strip()
    # Trim figure/chart noise that sometimes follows the abstract block.
    abstract = regex.split(r"\b0\.0\b|\bFigure\b", abstract, maxsplit=1)[0].strip()
    return abstract


def split_sections(full_text: str) -> dict[str, str]:
    """Map required and useful optional section keys to extracted text."""
    sections = {key: "" for key in CANONICAL_SECTIONS}
    sections["abstract"] = extract_abstract(full_text)

    headings = find_numbered_headings(full_text)
    if not headings:
        return sections

    blocks: dict[str, list[str]] = {
        key: []
        for key in (*CANONICAL_SECTIONS, *OPTIONAL_SECTIONS)
        if key != "abstract"
    }
    for index, (heading, start) in enumerate(headings):
        end = headings[index + 1][1] if index + 1 < len(headings) else len(full_text)
        body = full_text[start:end]
        body = regex.sub(r"(?m)^" + regex.escape(heading) + r"\s*", "", body, count=1).strip()
        key = _canonical_section_key(heading)
        if key and key != "abstract" and body:
            blocks[key].append(body)

    for key, parts in blocks.items():
        if parts:
            sections[key] = "\n\n".join(parts).strip()
    return sections


def _references_section_text(full_text: str) -> str:
    """Return bibliography body text after the references heading."""
    lower = full_text.lower()
    start = -1
    for heading in _REFERENCE_HEADINGS:
        match = regex.search(rf"(?m)^\s*\d*\s*{heading}\s*$", lower)
        if match:
            start = match.start()
            break
    if start < 0:
        match = regex.search(r"(?is)\breferences\b", full_text)
        if not match:
            return ""
        start = match.end()

    tail = full_text[start:]
    tail = regex.sub(r"(?is)\bappendix\b.*", "", tail).strip()
    tail = regex.sub(
        r"(?is)^\s*(?:\d+\s+)?(?:references|bibliography|works cited)\s*",
        "",
        tail,
        count=1,
    ).strip()
    return tail


def _normalize_reference_entry(entry: str) -> str:
    entry = regex.sub(r"\s+", " ", entry).strip(" ,;.")
    entry = regex.sub(r"^(?:\[\d{1,3}\]|\d{1,3}\.)\s+", "", entry)
    return entry.strip()


def _is_reference_entry(entry: str) -> bool:
    if len(entry) < 20:
        return False
    lowered = entry.lower()
    if lowered in _REFERENCE_HEADINGS:
        return False
    if regex.match(r"^\d{1,3}$", entry):
        return False
    return bool(regex.search(r"[A-Za-z]", entry))


def _split_numbered_references(tail: str) -> list[str]:
    """Split [n] and n. numbered bibliographies."""
    parts = regex.split(r"(?m)^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s+", tail)
    return [_normalize_reference_entry(part) for part in parts if _is_reference_entry(part)]


def _numbered_reference_marker_count(tail: str) -> int:
    return len(regex.findall(r"(?m)^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s+", tail))


def _reference_lines(tail: str) -> list[str]:
    """Keep bibliography line boundaries while removing PDF page-number noise."""
    lines: list[str] = []
    for raw_line in tail.splitlines():
        line = regex.sub(r"\s+", " ", raw_line).strip()
        if not line or regex.fullmatch(r"\d{1,3}", line):
            continue
        lines.append(line)
    return lines


def _looks_like_reference_start(line: str) -> bool:
    """Return whether a line begins with a plausible academic author list."""
    if not line or not regex.match(r"^[\p{Lu}]", line):
        return False
    if regex.search(r"^(?:In|Proceedings|International|Journal|Conference|University)\b", line):
        return False

    surname_initial = regex.match(
        r"^[\p{Lu}][\p{L}'’\-]+,\s+(?:[\p{Lu}]\.?(?:\s+[\p{Lu}]\.?)?)",
        line,
    )
    if surname_initial:
        return True

    if regex.search(r"^[^.!?]{1,120}\bet\s+al\.", line, regex.IGNORECASE):
        return True

    named_author = regex.match(
        r"^[\p{Lu}][\p{L}'’\-]+"
        r"(?:\s+(?:[\p{Lu}]\.|[\p{Lu}][\p{L}'’\-]+)){1,6}"
        r"(?:,|\s+and\b|\s+et\s+al\.|\.\s+)",
        line,
    )
    if named_author:
        return True

    author_prefix = regex.split(r"\.\s+(?=[\p{Lu}])", line, maxsplit=1)[0]
    if regex.search(r"\b(?:and|&)\b", author_prefix, regex.IGNORECASE):
        return True
    return line.count(",") >= 2


def _split_author_year_references(tail: str) -> list[str]:
    """Split unnumbered bibliographies without breaking wrapped citations."""
    entries: list[str] = []
    current: list[str] = []
    for line in _reference_lines(tail):
        current_text = " ".join(current)
        starts_new = (
            bool(current)
            and bool(_REFERENCE_YEAR_RE.search(current_text))
            and _looks_like_reference_start(line)
        )
        if starts_new:
            entry = _normalize_reference_entry(current_text)
            if _is_reference_entry(entry):
                entries.append(entry)
            current = []
        current.append(line)

    if current:
        entry = _normalize_reference_entry(" ".join(current))
        if _is_reference_entry(entry):
            entries.append(entry)
    return entries


def _reference_years(entry: str) -> list[str]:
    return _REFERENCE_YEAR_RE.findall(entry)


def _reference_split_score(refs: list[str]) -> float:
    """Prefer complete citations and penalize fragments or merged entries."""
    if not refs:
        return float("-inf")

    score = 0.0
    for ref in refs:
        length = len(ref)
        years = _reference_years(ref)
        has_author = _looks_like_reference_start(ref)
        if has_author:
            score += 2.0
        if years:
            score += 2.0
        if 40 <= length <= 450:
            score += 1.0
        if length < 30:
            score -= 4.0
        if length > 600:
            score -= 3.0 + (length - 600) / 200
        if len(years) > 2:
            score -= (len(years) - 2) * 2.0
        if not has_author or not years:
            score -= 2.0
    return score


def extract_references(full_text: str) -> list[str]:
    """Extract one clean citation string per bibliography entry."""
    tail = _references_section_text(full_text)
    if not tail:
        return []

    numbered = _split_numbered_references(tail)
    if _numbered_reference_marker_count(tail) >= 2:
        best = numbered
    else:
        candidates = (numbered, _split_author_year_references(tail))
        best = max(candidates, key=_reference_split_score)

    deduped: list[str] = []
    seen: set[str] = set()
    for ref in best:
        key = regex.sub(r"[\W_]+", " ", ref.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped[:200]


def _looks_like_author_line(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    if regex.search(r"\b(university|institute|fair|google|arxiv|http|@)\b", line, regex.IGNORECASE):
        return False
    if regex.search(r"\b(and|&)\b", line, regex.IGNORECASE):
        return True
    return bool(regex.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z'.-]+){0,4}$", line.strip()))


def _parse_authors(lines: list[str]) -> list[str]:
    authors: list[str] = []
    for line in lines:
        if not _looks_like_author_line(line):
            continue
        parts = regex.split(r"\s+and\s+|\s*&\s*", line, flags=regex.IGNORECASE)
        for part in parts:
            name = part.strip(" ,")
            if name and name not in authors:
                authors.append(name)
    return authors


def extract_first_page_metadata(page_text: str, pdf_metadata: dict[str, Any]) -> dict[str, Any]:
    """Populate title/authors/abstract hints from page one and PDF metadata."""
    lines = [regex.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
    lines = [line for line in lines if line]
    title_lines: list[str] = []
    author_lines: list[str] = []
    reached_abstract = False
    for line in lines:
        lower = line.lower()
        if lower.startswith("abstract"):
            reached_abstract = True
            break
        if regex.search(r"arxiv:\s*\d{4}\.\d+", line, regex.IGNORECASE):
            continue
        if _looks_like_author_line(line):
            author_lines.append(line)
            continue
        if not author_lines and not reached_abstract:
            if len(line) > 3 and not regex.match(r"^\d", line):
                title_lines.append(line)

    title = (pdf_metadata.get("/Title") or "").strip()
    if not title and title_lines:
        title = " ".join(title_lines[:3]).strip()
    authors = _parse_authors(author_lines)
    abstract_hint = extract_abstract(page_text)
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract_hint,
    }


def build_parsed_paper(pdf_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse a PDF into canonical ParsedPaper JSON and debug metadata."""
    pdf_path = pdf_path.resolve()
    page_texts, warnings = extract_page_texts(pdf_path)
    full_text = join_page_text(page_texts)
    reader = PdfReader(str(pdf_path))
    raw_meta = dict(reader.metadata or {})
    first_page_meta = extract_first_page_metadata(page_texts[0] if page_texts else "", raw_meta)
    sections = split_sections(full_text)
    references = extract_references(full_text)
    headings = [heading for heading, _ in find_numbered_headings(full_text)]

    paper_id = stable_paper_id(pdf_path)
    arxiv_id = _arxiv_id_from_name(pdf_path.name)
    metadata = {
        "paper_id": paper_id,
        "title": first_page_meta["title"],
        "abstract": sections["abstract"] or first_page_meta["abstract"],
        "authors": first_page_meta["authors"],
        "categories": [],
        "published_date": None,
        "venue": None,
        "doi": None,
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "source": "uploaded_pdf",
    }
    if not sections["abstract"]:
        sections["abstract"] = metadata["abstract"] or ""

    parsed = {
        "metadata": metadata,
        "sections": sections,
        "references": references,
        "keywords": [],
        "entities": dict(EMPTY_ENTITIES),
    }
    debug = {
        "pdf_path": str(pdf_path),
        "page_count": len(page_texts),
        "headings": headings,
        "warnings": warnings,
        "page_text": page_texts,
    }
    return parsed, debug
