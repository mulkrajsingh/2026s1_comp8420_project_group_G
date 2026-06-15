# Base vs LoRA Table

This table uses the selected base and adapter rows from `final_model_comparison.csv`.
Proxy rows are smoke-test evidence only; Ollama rows are required for final claims.

| Prompt | Base structure | Adapter structure | Base faithfulness | Adapter faithfulness | Base output | Adapter output |
| --- | --- | --- | --- | --- | --- | --- |
| P04 citation_recommendation | 0.81 | 0.81 | 1.0 | 1.0 | `results/model_comparison/ollama_20260614_citation_guard_enforced/model_outputs/P04_base.md` | `results/model_comparison/ollama_20260614_citation_guard_enforced/model_outputs/P04_lora.md` |
