# Base vs LoRA Table

This table uses the selected base and adapter rows from `final_model_comparison.csv`.
Proxy rows are smoke-test evidence only; Ollama rows are required for final claims.

| Prompt | Base structure | Adapter structure | Base faithfulness | Adapter faithfulness | Base output | Adapter output |
| --- | --- | --- | --- | --- | --- | --- |
| P01 uploaded_paper_summary | 0.63 | 0.63 | 0.5 | 0.5 | `results/model_comparison/ollama_20260614/model_outputs/P01_base.md` | `results/model_comparison/ollama_20260614/model_outputs/P01_lora.md` |
| P02 topic_search_synthesis | 0.9 | 0.81 | 1.0 | 1.0 | `results/model_comparison/ollama_20260614/model_outputs/P02_base.md` | `results/model_comparison/ollama_20260614/model_outputs/P02_lora.md` |
| P03 research_gap_identification | 0.81 | 0.81 | 0.5 | 0.5 | `results/model_comparison/ollama_20260614/model_outputs/P03_base.md` | `results/model_comparison/ollama_20260614/model_outputs/P03_lora.md` |
| P04 citation_recommendation | 0.63 | 0.63 | 0.5 | 0.5 | `results/model_comparison/ollama_20260614/model_outputs/P04_base.md` | `results/model_comparison/ollama_20260614/model_outputs/P04_lora.md` |
| P05 peer_review_critique | 0.9 | 0.9 | 0.5 | 0.5 | `results/model_comparison/ollama_20260614/model_outputs/P05_base.md` | `results/model_comparison/ollama_20260614/model_outputs/P05_lora.md` |
| P06 beginner_explanation | 0.72 | 0.63 | 0.5 | 0.5 | `results/model_comparison/ollama_20260614/model_outputs/P06_base.md` | `results/model_comparison/ollama_20260614/model_outputs/P06_lora.md` |
