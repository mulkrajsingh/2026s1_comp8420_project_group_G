#!/usr/bin/env bash
# Thin wrapper for markers who prefer bash.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec python -m lora_dataset.create_dataset "$@"
