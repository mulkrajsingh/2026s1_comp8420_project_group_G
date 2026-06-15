# Qwen3 8B QLoRA Training Run

## Release

- Run key: `qwen3_8b_lora_v2`
- Completed checkpoint: `checkpoint-1730`
- Adapter archive: `models/releases/research_lora_adapter_20260609_114209.zip`
- Archive size: 26,248,547 bytes
- Archive SHA-256: `7eac8f735b93e4ddb1d78613924a5f409eed88de878c7212a155ddd8b7dd3a59`
- Adapter weights SHA-256: `0a8a0403740c0343a5d3157c68fedfb8a35e4ea4a343b8bbd142f6e12129d638`

## Training

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-8B` |
| Method | QLoRA with PEFT 0.19.1 and TRL |
| Dataset | 17,298 chat-format rows |
| Split | 80/10/10, seed 42 |
| Epochs | 1 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4 |
| Maximum sequence length | 4096 |
| Effective batch size | 8 |
| Checkpoint interval | 50 optimizer steps |
| Runtime environment | Google Colab |

The exact GPU model, wall-clock runtime, and carbon estimate were not retained
in the exported artifacts and must not be invented in the report.

## Split Metrics

| Split | Loss | Perplexity | Mean token accuracy | Evaluated rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.9326 | 2.5412 | 0.7910 | 500 |
| Validation | 0.9388 | 2.5570 | 0.7900 | 500 |
| Test | 0.9446 | 2.5717 | 0.7882 | 500 |

These are training-loss and token-accuracy measurements. They do not establish
better generated-answer quality than the base model. That claim requires the
separate Ollama base-versus-LoRA comparison.
