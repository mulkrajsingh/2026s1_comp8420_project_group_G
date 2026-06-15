"""Stage 05 — frontend notes generator.

The canonical Vite UI is under ``frontend/`` and its production build is served
by the FastAPI application.
"""
from __future__ import annotations

from .io_paths import write_text


def write_frontend_notes() -> str:
    md = """# Frontend demo notes (Stage 05)

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

## Design intent
Refined-minimal, editorial/academic tone (system serif display + clean sans body).
Clear and useful over decorative, as required by the team plan.
"""
    return write_text("frontend_demo_notes.md", md)
