#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m ruff check .
python -m vulture \
  integration/app \
  modules/dataset \
  modules/pdf_nlp \
  modules/retrieval/app \
  modules/llm/app \
  --min-confidence 90
