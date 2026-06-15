"""APA-style citation formatting for PaperRecord objects.

Formats author lists, year, title, venue, and DOI or arXiv URL following
APA 7th edition conventions used in recommendation exports.
"""

from __future__ import annotations

import re


def _format_author(name: str) -> str:
    """Convert 'Firstname Lastname' or 'F. Lastname' to APA 'Lastname, F.' format."""
    name = name.strip()
    if not name:
        return "Unknown"
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    # Last word is the surname; all preceding parts are initials
    surname = parts[-1]
    initials = " ".join(p[0].upper() + "." for p in parts[:-1] if p)
    return f"{surname}, {initials}"


def format_apa(paper: dict) -> str:
    """Return a full APA 7th edition citation string for a PaperRecord."""
    authors = paper.get("authors", [])
    year = (paper.get("published_date") or "")[:4] or "n.d."
    title = paper.get("title", "Untitled")
    venue = paper.get("venue") or "arXiv"
    doi = paper.get("doi")
    url = paper.get("url")
    arxiv_id = paper.get("arxiv_id")

    # Format author list per APA rules
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        author_str = _format_author(authors[0])
    elif len(authors) <= 20:
        formatted = [_format_author(a) for a in authors]
        author_str = ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    else:
        # APA: list first 19, ellipsis, then final author
        formatted = [_format_author(a) for a in authors[:19]]
        author_str = ", ".join(formatted) + f", . . . {_format_author(authors[-1])}"

    citation = f"{author_str} ({year}). {title}. *{venue}*."

    if doi:
        # Normalise — strip any existing prefix
        doi_clean = re.sub(r"^https?://doi\.org/", "", doi)
        citation += f" https://doi.org/{doi_clean}"
    elif arxiv_id:
        citation += f" https://arxiv.org/abs/{arxiv_id}"
    elif url:
        citation += f" {url}"

    return citation


def format_apa_list(papers: list[dict], start: int = 1) -> str:
    """Return a numbered APA reference list as a markdown string."""
    lines = []
    for i, paper in enumerate(papers, start=start):
        lines.append(f"{i}. {format_apa(paper)}")
    return "\n".join(lines)
