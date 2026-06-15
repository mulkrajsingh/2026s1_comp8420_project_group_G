# Local Runtime Documentation

## Backend

Generation uses **Ollama** on localhost. There is no alternate deterministic generation backend in this workstream.

| Backend | Purpose | Requires |
| --- | --- | --- |
| `ollama` | local model output for report claims | Ollama running on localhost |

## Models

| Role | Ollama tag | Hugging Face |
| --- | --- | --- |
| Base | `qwen3:8b` | `Qwen/Qwen3-8B` |
| Team LoRA/QLoRA (after merge) | `qwen3-research-lora:latest` | PEFT under `models/adapters/research_lora_adapter/` |

There is no alternate fallback model family in this workstream. Optional rubric evidence
can compare **quantization variants of the same** `qwen3:8b` tag (Q8, Q6, Q4), not other
model sizes.

## Quantization Comparison

Run the same fixed prompts for every available variant of `qwen3:8b`:

- full precision or FP16/HF where feasible
- Q8
- Q6
- Q4
- team-trained adapter tag `qwen3-research-lora:latest`

Track latency, error status, output path, structure compliance, evidence faithfulness,
citation safety, and human spot-check notes.

## Commands

```bash
ollama pull qwen3:8b

python -m app.cli analyze-query --query "Summarise BM25 briefly"
python -m app.cli summarize --model qwen3:8b --paper outputs/parsed_paper.json --out outputs/paper_summary.md
python -m app.cli synthesize --evidence outputs/rag_evidence_pack.json --out outputs/llm_analysis.md
python -m app.cli compare-models --models base=qwen3:8b,lora=qwen3-research-lora:latest --test-set data/eval/fixed_prompts.jsonl --out results/model_comparison/
```

`summarize` uses only the supplied `ParsedPaper`; it does not load a
`RagEvidencePack` or perform retrieval. Use `synthesize --evidence ...` for
retrieved-evidence and related-paper analysis.

Both commands default to `--style auto`. Their generation metadata records the
structured query analysis, requested style, and resolved style. Pass an explicit
non-auto style to override classification.

Generation metadata also records the input contract and whether retrieval/external
evidence was used. Production commands never create sample inputs when files are
missing.

If Ollama is unavailable, `summarize` writes the backend error to its Markdown and
metadata outputs, then exits nonzero. Comparison commands record the error in their
output rows rather than pretending a model run succeeded.

## Build the Ollama Adapter Tag

The active trained PEFT adapter lives under
`models/adapters/research_lora_adapter/`. Its source archive is
`models/releases/research_lora_adapter_20260609_114209.zip`; hashes and the
retained June 9 training metrics are documented in
`results/model_comparison/lora_training_run_20260609.md`. Generated GGUF and
merged outputs remain local artifacts.

Prerequisites:

- Ollama installed and available on `PATH`
- Python packages from the repository root `requirements.txt`, including
  `torch`, `transformers`, `peft`, and `sentencepiece`
- a local llama.cpp checkout, supplied with `--llama-cpp` or `LLAMA_CPP_ROOT`

From `modules/llm/`:

```bash
python scripts/build_ollama_research_lora_model.py --check-only

python scripts/build_ollama_research_lora_model.py \
  --llama-cpp /path/to/llama.cpp

ollama list
```

By default, the script uses the low-memory adapter path: it converts only the PEFT LoRA
adapter to GGUF, writes a Modelfile with `FROM qwen3:8b` and `ADAPTER`, and runs
`ollama create qwen3-research-lora:latest`. This avoids creating a 15+ GB merged F16
GGUF. The old full-merge path remains available with `--merge-mode full` for machines
with enough RAM and disk. The script does not auto-clone llama.cpp or switch to another
model family.

After the tag exists, regenerate rubric-grade comparison rows with:

```bash
python -m app.cli compare-models \
  --models base=qwen3:8b,lora=qwen3-research-lora:latest \
  --test-set data/eval/fixed_prompts.jsonl \
  --out results/model_comparison/
```
