#!/usr/bin/env bash
# Real-paper system tests — module + integration E2E (Ollama required for LLM tests).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/integration"

echo "=== Bootstrap test artifacts (parse PDFs, retrieval pack) ==="
python tests/harness/bootstrap_artifacts.py

if [[ "${SKIP_OLLAMA:-0}" == "1" ]]; then
  python -m unittest discover -s tests/e2e -p 'test_pdf_nlp.py' -v
  python -m unittest discover -s tests/e2e -p 'test_retrieval.py' -v
else
  if ! command -v ollama >/dev/null 2>&1; then
    if [[ "${REQUIRE_OLLAMA:-0}" == "1" ]]; then
      echo "Ollama required but not installed." >&2
      exit 1
    fi
    echo "Ollama not found — skipping LLM and integration E2E tests"
    python -m unittest discover -s tests/e2e -p 'test_pdf_nlp.py' -v
    python -m unittest discover -s tests/e2e -p 'test_retrieval.py' -v
    exit 0
  fi

  echo "=== E2E tests (all) ==="
  python -m unittest discover -s tests/e2e -p 'test_*.py' -v
fi

echo "=== Module unit tests ==="
(cd modules/pdf_nlp && python -m unittest discover -s tests -p 'test_*.py' -v)
(cd modules/llm && python -m unittest discover -s tests -p 'test_*.py' -v)
(cd integration && PYTHONPATH="${ROOT}:${ROOT}/integration" python -m unittest discover -s tests -p 'test_*.py' -v)

echo "System tests complete."
