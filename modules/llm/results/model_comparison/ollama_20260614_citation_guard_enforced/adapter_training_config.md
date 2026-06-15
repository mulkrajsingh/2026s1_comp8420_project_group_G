# Adapter Training Config

| Field | Planned value | Notes |
| --- | --- | --- |
| Base model | Qwen/Qwen3-8B | matches Ollama `qwen3:8b` |
| Method | QLoRA via PEFT/TRL or Unsloth | Colab/RunPod recommended |
| LoRA rank | 16 | start conservative |
| LoRA alpha | 32 | standard 2x rank starting point |
| Dropout | 0.05 | reduce overfitting |
| Epochs | 1-3 | stop early on format regression |
| Learning rate | 2e-4 | tune if outputs become verbose or unstable |
| Max sequence length | 4096-8192 | depends on GPU memory |
| Target modules | q_proj, k_proj, v_proj, o_proj | verify per base model |
| Output path | models/adapters/research_lora_adapter/ | do not commit large weights |

Real training logs, exact dataset size, and adapter checksum must be added after the
training run. This file is the reproducible workflow contract.
