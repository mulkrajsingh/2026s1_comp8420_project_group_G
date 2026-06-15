# Runtime Notes

Current backend used for this artifact set: `ollama`.

Ollama calls a local daemon and records latency, model tag, prompt id, error
status, and source IDs used.

For final report claims, use measured Ollama outputs for the fixed prompts.

| Variant | Ollama tag | Role | Quantization | Adapter | Notes |
| --- | --- | --- | --- | --- | --- |
| `qwen3_8b` | `qwen3:8b` | local foundation model | fp16_or_q8_when_available | none | Ollama qwen3:8b; HF Qwen/Qwen3-8B for training. |
| `qwen3_8b_lora` | `qwen3-research-lora:latest` | team-trained LoRA/QLoRA adapter | qlora_4bit_training_then_adapter_inference | models/adapters/research_lora_adapter/ | Merge adapter into qwen3:8b and deploy as qwen3-research-lora:latest. |

Required real-run measurements: latency, memory notes, output path, quantization
level, structure compliance, evidence faithfulness, and human spot-check notes.
