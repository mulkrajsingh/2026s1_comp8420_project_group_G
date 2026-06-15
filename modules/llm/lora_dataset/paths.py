"""Paths for LoRA dataset artifacts (relative to modules/llm root)."""

from __future__ import annotations

from pathlib import Path

WORKSTREAM_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = WORKSTREAM_ROOT / "data" / "processed"
DATA_RAW = WORKSTREAM_ROOT / "data" / "raw"

KAGGLE_RAW = DATA_RAW / "arxiv-metadata-oai-snapshot.json"
HYBRID_JSONL = DATA_PROCESSED / "open_academic_researchqa_hybrid_lora_dataset.jsonl"
HYBRID_MANIFEST = DATA_PROCESSED / "open_academic_researchqa_hybrid_lora_dataset_manifest.md"
CORPUS_JSONL = DATA_PROCESSED / "project_arxiv_rag_corpus_3k.jsonl"
CORPUS_MANIFEST = DATA_PROCESSED / "project_arxiv_rag_corpus_3k_manifest.md"
RAG_JSONL = DATA_PROCESSED / "project_arxiv_rag_lora_instructions.jsonl"
RAG_QUERIES_JSONL = DATA_PROCESSED / "project_arxiv_rag_queries.jsonl"
RAG_PACKS_DIR = DATA_PROCESSED / "project_arxiv_rag_packs"
PAPER_ONLY_JSONL = DATA_PROCESSED / "project_paper_only_lora_instructions.jsonl"
FINAL_DATASET_DIR = DATA_PROCESSED / "final_dataset"
TRAIN_JSONL = FINAL_DATASET_DIR / "research_lora_train.jsonl"
TRAIN_MANIFEST = FINAL_DATASET_DIR / "research_lora_train_manifest.md"
TRAIN_ZIP = FINAL_DATASET_DIR / "research_lora_train.zip"

SEED = 42
OPEN_ACADEMIC_PER_SOURCE = 3000
RESEARCHQA_LIMIT = 3000
CORPUS_LIMIT = 3000
RAG_NUM_QUERIES = 500
PAPER_ONLY_PER_TASK = 100
MAX_TRAINING_PROMPT_CHARS = 14_000
