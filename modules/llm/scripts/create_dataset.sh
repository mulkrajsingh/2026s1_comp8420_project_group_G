#!/usr/bin/env bash
# Shell entry point for the LoRA dataset builder in lora_dataset/create_dataset.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec python -m lora_dataset.create_dataset "$@"
