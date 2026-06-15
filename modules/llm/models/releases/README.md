# LoRA Adapter Releases

The canonical release is:

`research_lora_adapter_20260609_114209.zip`

It contains the PEFT adapter produced by the completed `qwen3_8b_lora_v2`
training run. Extract it under `models/adapters/` so the runtime path is:

`models/adapters/research_lora_adapter/`

## Integrity

- Archive SHA-256: `7eac8f735b93e4ddb1d78613924a5f409eed88de878c7212a155ddd8b7dd3a59`
- Adapter weights SHA-256: `0a8a0403740c0343a5d3157c68fedfb8a35e4ea4a343b8bbd142f6e12129d638`
- Adapter config SHA-256: `029be66acfc82368910380a73a52881470748ef6a55b71ccffc11eeda5cbd1cc`
- Converted adapter GGUF SHA-256: `d40d0b7415f62e1de3f01a8e18b0e57433681fc5e72170cc44416558b996b303`

Do not add older dated archives beside this release. Evaluation and training
details are recorded in
`results/model_comparison/lora_training_run_20260609.md`.
