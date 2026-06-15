"""Generator for frontend demo notes under ``outputs/``.

The Vite UI lives in ``frontend/``; its production build is served by the FastAPI
application at the repository root URL.
"""
from __future__ import annotations

from .io_paths import write_text


def write_frontend_notes() -> str:
    """Write ``outputs/frontend_demo_notes.md`` describing the web UI."""
    md = """# Frontend demo notes

Vite frontend: `frontend/`. It uses no external runtime CDN and talks to the
same-origin FastAPI application.

## Two input modes
- **PDF upload** -> `POST /api/analyze-pdf`
- **Topic search** -> `POST /api/search-topic`

## Results panel renders the full AnalysisResult
metadata, summary (AI-disclosed), key findings, research gaps, recommended papers,
APA citations, evidence snippets, peer-review feedback.

## Run
```bash
# production UI and API from the repository root
scripts/rpa web

# frontend development with Vite hot reload
scripts/rpa web --reload
cd integration/frontend && pnpm dev
```
If the API is not running, the page shows a clear "start the API" message rather
than crashing.

## Design
Refined-minimal, editorial tone (serif display + sans body). Layout prioritises
readability of long analysis sections over decorative chrome.
"""
    return write_text("frontend_demo_notes.md", md)
