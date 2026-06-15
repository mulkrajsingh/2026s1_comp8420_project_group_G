# Adapter Training Config

| Field | Completed value | Notes |
| --- | --- | --- |
| Base model | Qwen/Qwen3-8B | matches Ollama `qwen3:8b` |
| Method | QLoRA via PEFT 0.19.1 / TRL | Google Colab |
| LoRA rank | 16 | completed run |
| LoRA alpha | 32 | completed run |
| Dropout | 0.05 | completed run |
| Epochs | 1 | 1,730 optimizer steps |
| Learning rate | 2e-4 | completed run |
| Max sequence length | 4096 | completed run |
| Effective batch size | 8 | batch 1, accumulation 8 |
| Target modules | q_proj, k_proj, v_proj, o_proj | adapter config |
| Runtime path | models/adapters/research_lora_adapter/ | extracted local adapter |
| Release path | models/releases/research_lora_adapter_20260609_114209.zip | tracked archive |

See `lora_training_run_20260609.md` for split metrics, checksums, dataset
details, and limitations.
