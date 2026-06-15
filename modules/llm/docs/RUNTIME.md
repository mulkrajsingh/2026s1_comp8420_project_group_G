# LLM Runtime

Local generation uses **Ollama** on localhost. There is no alternate
deterministic generation backend in this module.

| Backend | Purpose | Requires |
| --- | --- | --- |
| `ollama` | Local model output | Ollama running on localhost |

## Models

| Role | Ollama tag | Hugging Face |
| --- | --- | --- |
| Base | `qwen3:8b` | `Qwen/Qwen3-8B` |
| Team LoRA/QLoRA (after deployment) | `qwen3-research-lora:latest` | PEFT under `models/adapters/research_lora_adapter/` |

The project uses a single model family. Optional evaluation can compare
**quantization variants** of `qwen3:8b` (Q8, Q6, Q4), not other model sizes.

## Commands

```bash
ollama pull qwen3:8b

python -m app.cli analyze-query --query "Summarise BM25 briefly"
python -m app.cli summarize --model qwen3:8b --paper <parsed-paper.json> --out outputs/paper_summary.md
python -m app.cli synthesize --evidence <rag_evidence_pack.json> --out outputs/llm_analysis.md
python -m app.cli compare-models --models base=qwen3:8b,lora=qwen3-research-lora:latest \
  --test-set data/eval/fixed_prompts.jsonl --out results/model_comparison/
```

`summarize` uses only the supplied `ParsedPaper`; it does not load a
`RagEvidencePack` or perform retrieval. Use `synthesize --evidence ...` for
retrieved-evidence and related-paper analysis.

Both commands default to `--style auto`. Generation metadata records the
structured query analysis, requested style, and resolved style.

During end-to-end runs, outputs are written to `integration/outputs/`, not
`modules/llm/outputs/`.

If Ollama is unavailable, `summarize` writes the backend error to its Markdown
and metadata outputs, then exits nonzero.

## Build the Ollama adapter tag

The trained PEFT adapter is delivered via `setup_assets.py` to
`models/adapters/research_lora_adapter/`. The source archive and training
metrics are documented in
`results/model_comparison/lora_training_run_20260609.md`.

Prerequisites:

- Ollama installed and available on `PATH`
- Python packages from the repository root `requirements.txt`
- A local llama.cpp checkout, supplied with `--llama-cpp` or `LLAMA_CPP_ROOT`

From `modules/llm/`:

```bash
python scripts/build_ollama_research_lora_model.py --check-only

python scripts/build_ollama_research_lora_model.py \
  --llama-cpp /path/to/llama.cpp

ollama list
```

By default, the script converts only the PEFT LoRA adapter to GGUF, writes a
Modelfile with `FROM qwen3:8b` and `ADAPTER`, and runs
`ollama create qwen3-research-lora:latest`. The full merged F16 GGUF path is
available with `--merge-mode full` for machines with sufficient RAM and disk.

After the tag exists, regenerate comparison rows with:

```bash
python -m app.cli compare-models \
  --models base=qwen3:8b,lora=qwen3-research-lora:latest \
  --test-set data/eval/fixed_prompts.jsonl \
  --out results/model_comparison/
```

## Quantization comparison

Run the same fixed prompts for every available variant of `qwen3:8b`:

- full precision or FP16/HF where feasible
- Q8, Q6, Q4
- team-trained adapter tag `qwen3-research-lora:latest`

Track latency, error status, structure compliance, evidence faithfulness,
citation safety, and human spot-check notes.
