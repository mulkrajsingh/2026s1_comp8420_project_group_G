"""
Create the full research LoRA training dataset.

Run from modules/llm/:
  python -m lora_dataset.create_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure workstream root is on sys.path when run as script.
_WORKSTREAM = Path(__file__).resolve().parent.parent
if str(_WORKSTREAM) not in sys.path:
    sys.path.insert(0, str(_WORKSTREAM))

from lora_dataset.arxiv_rag import build_rag_instructions
from lora_dataset.env import load_dotenv
from lora_dataset.kaggle_corpus import build_corpus
from lora_dataset.kaggle_corpus import normalize_corpus_dates
from lora_dataset.kaggle_download import download_kaggle_raw, _sha256
from lora_dataset.merge_train import merge_train
from lora_dataset.open_hybrid import build_open_hybrid
from lora_dataset.paper_only import build_paper_only_instructions
from lora_dataset.paths import CORPUS_JSONL, CORPUS_MANIFEST, HYBRID_JSONL, KAGGLE_RAW, TRAIN_JSONL, TRAIN_ZIP


def _reject_synthetic_corpus() -> None:
    if not CORPUS_JSONL.is_file():
        return
    first = CORPUS_JSONL.read_text(encoding="utf-8").splitlines()[:1]
    if first and '"paper_id": "sample_' in first[0]:
        raise SystemExit(
            f"ERROR: {CORPUS_JSONL} is synthetic. Delete it and re-run create_dataset."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full research LoRA training dataset.")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip HF open + ResearchQA step.")
    parser.add_argument("--skip-download", action="store_true", help="Require existing Kaggle raw file.")
    parser.add_argument("--rebuild-corpus", action="store_true", help="Resample 3k PaperRecords.")
    parser.add_argument("--no-seeds", action="store_true", help="Omit 6 local_fixed_prompt rows in merge.")
    args = parser.parse_args()

    loaded = load_dotenv()
    if loaded:
        print(f"Loaded environment from {loaded}")

    print("=== Research LoRA dataset pipeline ===\n")

    if not args.skip_hybrid:
        print("--- Open academic + ResearchQA hybrid ---")
        build_open_hybrid()
    elif not HYBRID_JSONL.is_file():
        raise SystemExit(f"Missing {HYBRID_JSONL}; run without --skip-hybrid.")

    if args.skip_download and not KAGGLE_RAW.is_file():
        raise SystemExit(f"Missing {KAGGLE_RAW} and --skip-download was set.")

    if not args.skip_download and not KAGGLE_RAW.is_file():
        print("--- Kaggle arXiv metadata download ---")
        download_kaggle_raw()
    elif KAGGLE_RAW.is_file():
        print(f"--- Kaggle raw present ({KAGGLE_RAW}) ---")

    _reject_synthetic_corpus()
    if args.rebuild_corpus:
        CORPUS_JSONL.unlink(missing_ok=True)
        CORPUS_MANIFEST.unlink(missing_ok=True)

    if not CORPUS_JSONL.is_file():
        print("--- Project arXiv RAG corpus (random 3k) ---")
        build_corpus()
    else:
        print(f"--- Corpus present ({CORPUS_JSONL}) ---")

    print("--- Normalize corpus dates ---")
    normalize_corpus_dates()

    print("--- Project paper-only instructions ---")
    build_paper_only_instructions()

    print("--- Project arXiv RAG instructions + merge ---")
    build_rag_instructions()
    counts = merge_train(include_seeds=not args.no_seeds)

    print(f"\nDone. Train JSONL: {TRAIN_JSONL}")
    if TRAIN_ZIP.is_file():
        print(f"Train zip: {TRAIN_ZIP}")
    print("Rows by source:", counts)
    if KAGGLE_RAW.is_file():
        print(f"Kaggle raw SHA256: {_sha256(KAGGLE_RAW)}")


if __name__ == "__main__":
    main()
