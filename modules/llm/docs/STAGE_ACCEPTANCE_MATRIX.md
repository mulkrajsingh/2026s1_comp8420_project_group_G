# Stage Acceptance Matrix

| Stage | Command | Required outputs | Status |
| --- | --- | --- | --- |
| 01 | `compare-models` | `data/eval/fixed_prompts.jsonl`, retained `ollama_20260614/runtime_notes.md` | implemented |
| 02 | `synthesize`, `compare-prompts` | `outputs/llm_analysis.md`, `app/prompt_library.py` | implemented |
| 03 | `compare-prompts` | regenerate `results/prompt_comparison/` with Ollama | implemented |
| 04 | `chat` (topic RAG), `compare-prompts` | `app/react_loop.py`, integration retrieval chat wiring | implemented |
| 05 | `synthesize`, `summarize`, `review`, `training/evaluate_faithfulness.py` | `app/verification.py`, faithfulness audit in markdown | implemented |
| 06 | `python -m lora_dataset.create_dataset` | `data/processed/final_dataset/research_lora_train.jsonl`, `.zip`, manifest | implemented |
| 07 | `compare-models`, `notebooks/train_lora_adapter.ipynb` | `results/model_comparison/adapter_training_config.md`, `models/adapters/README.md` | implemented |
| 08 | `compare-models` | retained `ollama_20260614/base_vs_lora_table.md` and human review | implemented |
| 09 | — | removed; same-model self-verification used instead | removed |
| 10 | `synthesize` | `outputs/llm_analysis.md`, `outputs/llm_handoff.md`, `outputs/llm_generation.json` | implemented |
| 11 | `chat`, `summarize`, `analyze-query` | direct-chat and paper-only outputs, structured query analysis, retrieval-aware routing tests | implemented |

## Acceptance Caveat

The current status means the CLI commands pass and create the expected stage
evidence when run with Ollama. Final model-quality claims require Ollama rows
with empty error fields and human spot-check notes.

The test suite also verifies that missing production inputs fail clearly, paper-only
summaries never load RAG fixtures, explicit styles override inference, and integration
conversational chat uses the local synthesizer without loading retrieval providers.
