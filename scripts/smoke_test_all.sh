#!/usr/bin/env bash
# Quick compile and CLI smoke checks across integration and member modules.
# Exits non-zero on the first failing step.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
log_ok() { echo "[PASS] $*"; PASS=$((PASS + 1)); }
log_fail() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }

run() {
  local name="$1"
  shift
  echo ""
  echo "=== $name ==="
  if "$@"; then
    log_ok "$name"
  else
    log_fail "$name (exit $?)"
    return 1
  fi
}

# --- Integration ---
run "integration: compileall" bash -c 'cd integration && python -m compileall -q app'
run "integration: integration-status" bash -c 'cd integration && python -m app.cli integration-status'

# --- Repo root launcher ---
run "scripts/rpa integration-status" bash -c './scripts/rpa integration-status'

# --- modules/dataset ---
run "dataset: dev_5k.jsonl valid JSONL" python -c "
import json
from pathlib import Path
p = Path('modules/dataset/data/processed/dev_5k.jsonl')
n = 0
with p.open() as f:
    for line in f:
        json.loads(line)
        n += 1
assert n == 5000, n
print('lines', n)
"
run "dataset: symlink from integration" test -f integration/data/processed/dev_5k.jsonl

# --- modules/retrieval ---
run "retrieval: compileall" bash -c 'cd modules/retrieval && python -m compileall -q app'
run "retrieval: recommend-topic (tfidf, sample corpus)" bash -c '
cd modules/retrieval
python -m app.cli recommend-topic \
  --query "retrieval augmented generation" \
  --papers ../dataset/data/processed/dev_sample.jsonl \
  --out outputs/smoke_recommendations.json \
  --top-k 5 \
  --retrieval-strategy tfidf
test -s outputs/smoke_recommendations.json
'

# --- modules/pdf_nlp ---
run "pdf_nlp: import PdfParser + parse sample PDF" python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'modules/pdf_nlp')
from pdf_parser import PdfParser
sample = Path('tests/papers/drq_v2/2107.09645v1.pdf')
p = PdfParser(str(sample))
assert p.text and len(p.text) > 100, 'empty text'
print('chars', len(p.text), 'sections', len(p.sections))
"

# --- System tests (bootstrap + E2E; Ollama tests skipped if unavailable) ---
run "tests: run_system_tests.sh" bash -c 'chmod +x tests/run_system_tests.sh && SKIP_OLLAMA="${SKIP_OLLAMA:-0}" tests/run_system_tests.sh'

echo ""
echo "========================================"
echo "Smoke summary: $PASS passed, $FAIL failed"
echo "========================================"
test "$FAIL" -eq 0
