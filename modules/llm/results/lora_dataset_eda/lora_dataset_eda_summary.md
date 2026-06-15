# LoRA Training Dataset EDA

Generated from `data/processed/final_dataset/research_lora_train.jsonl`.

## Validation

- Rows parsed: 16,998
- Malformed JSON rows: 0
- Rows with required `system`, `user`, `assistant` message roles: 16,998
- Distinct sources: 7
- Distinct task labels: 16

## Source Mix

| Source | Rows | Share |
| --- | ---: | ---: |
| `project_arxiv_rag` | 3,000 | 17.6% |
| `researchqa` | 3,000 | 17.6% |
| `qasper` | 3,000 | 17.6% |
| `peerread` | 3,000 | 17.6% |
| `scicite` | 3,000 | 17.6% |
| `scitldr` | 1,992 | 11.7% |
| `local_fixed_prompt` | 6 | 0.0% |

## Task Mix

| Task | Rows |
| --- | ---: |
| `peer_review_critique` | 3,001 |
| `evidence_to_answer` | 3,000 |
| `citation_intent` | 3,000 |
| `paper_to_summary` | 1,992 |
| `lookup` | 976 |
| `comprehension` | 966 |
| `refusal` | 594 |
| `beginner_explanation` | 501 |
| `topic_search_synthesis` | 501 |
| `citation_recommendation` | 501 |
| `research_gap_identification` | 501 |
| `explain_recommendation` | 500 |
| `follow_up_qa` | 500 |
| `multi_hop` | 459 |
| `adversarial` | 5 |
| `uploaded_paper_summary` | 1 |

## Length And Evidence Coverage

- Median total row length: 210 words.
- P95 total row length: 2,223 words, driven by longer project RAG prompts.
- Rows with `[S#]` source-ID markers: 7,910 (46.5%).
- Rows with explicit evidence-passage blocks: 2,406 (14.2%).
- Project-aligned rows: 3,006 (`project_arxiv_rag` plus `local_fixed_prompt`).

## PPT Readout

The dataset combines broad academic supervision with project-specific RAG behavior. This supports a careful claim that the adapter training data is intentionally mixed for research-assistant structure, evidence grounding, citation behavior, and scientific summarization. It does not by itself prove adapter quality; model-quality claims still require real training and evaluation logs.
