"""Run prompt and model comparison from the training namespace.

Examples::

    python training/evaluate.py
    python training/evaluate.py --models base=qwen3:8b,lora=qwen3-research-lora:latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation import compare_models, compare_prompts, ensure_fixed_prompts, parse_model_specs  # noqa: E402
from app.runtime import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    GenerationConfig,
)


def main() -> None:
    """Run fixed-prompt comparisons for prompts and model variants."""
    parser = argparse.ArgumentParser(description="Evaluate Mulkraj prompt/model configurations.")
    parser.add_argument("--test-set", default="data/eval/fixed_prompts.jsonl")
    parser.add_argument("--prompt-out", default="results/prompt_comparison")
    parser.add_argument("--model-out", default="results/model_comparison")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--models", default=None)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    config = GenerationConfig(args.temperature, args.max_new_tokens, args.top_p)
    test_set = Path(args.test_set)
    ensure_fixed_prompts(test_set)
    compare_prompts(
        test_set,
        Path(args.prompt_out),
        model=args.model,
        ollama_host=args.ollama_host,
        config=config,
    )
    compare_models(
        test_set,
        Path(args.model_out),
        model_specs=parse_model_specs(args.models),
        ollama_host=args.ollama_host,
        config=config,
    )
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
