#!/usr/bin/env bash
# Record presentation POC artifacts (topic mode, real data, session JSONL).
# Prerequisites: Ollama running, `ollama pull qwen3:8b`, Bank + Mulkraj deps installed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SID="$ROOT/integration"
CORPUS="${CORPUS:-$ROOT/modules/dataset/data/processed/dev_5k.jsonl}"
QUERY="${QUERY:-retrieval augmented generation for scientific literature}"
LIMIT="${CORPUS_LIMIT:-1000}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "error: ollama not found in PATH" >&2
  exit 1
fi

echo "==> Preflight: ollama list (expect qwen3:8b)"
ollama list | head -20

cd "$SID"
pip install -q -r "$ROOT/requirements.txt" 2>/dev/null || true

echo "==> Running real-data POC (corpus limit=$LIMIT)"
python -m app.cli run \
  --query "$QUERY" \
  --corpus "$CORPUS" \
  --corpus-limit "$LIMIT" \
  --retrieval-strategy tfidf \
  --llm-model qwen3:8b

echo ""
echo "Artifacts:"
ls -la outputs/analysis_result.json outputs/demo_report.md outputs/demo_trace.json \
  outputs/llm_analysis.md outputs/llm_generation.json 2>/dev/null || true
echo "Latest session:"
ls -t data/sessions/session-*.jsonl 2>/dev/null | head -1
