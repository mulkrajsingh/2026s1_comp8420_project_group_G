# LoRA Training Dataset — Hybrid + arXiv RAG Manifest

Total training rows: 16998
Output: `data/processed/lora_train_hybrid_arxiv_rag.jsonl`
Shuffle seed: 42
SHA-256: `cf8642c4b61cf2fa5e75d86621be21daab34e13fc1e81612c3637d05e667d6ee`

The generated JSONL is intentionally not tracked. Rebuild it from the inputs
below; the checksum verifies that the canonical 16,998-row artifact was
reproduced.

## Inputs
- local_fixed_prompt (6)
- hybrid_open_data: open_academic_researchqa_hybrid_lora_dataset.jsonl (13992 rows)
- project_arxiv_rag: project_arxiv_rag_lora_instructions.jsonl (3000 rows)

Input checksums:

- `open_academic_researchqa_hybrid_lora_dataset.jsonl`:
  `503100bd6201471a9843949dfcd0b9dfda2b4845d1b2d65bb8a394cc18a0dbc1`
- `project_arxiv_rag_lora_instructions.jsonl`:
  `cfc57f5592da02b5f30d0aaf04fd2451434bb33543c45728594286f355533555`

## Rows by source
- `local_fixed_prompt`: 6
- `peerread`: 3000
- `project_arxiv_rag`: 3000
- `qasper`: 3000
- `researchqa`: 3000
- `scicite`: 3000
- `scitldr`: 1992

Train with the QLoRA notebook: `modules/llm/notebooks/train_lora_adapter.ipynb`.
