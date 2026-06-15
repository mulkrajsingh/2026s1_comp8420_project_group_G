# LLM Module

**Contributions:** [`docs/CONTRIBUTIONS.md`](../../docs/CONTRIBUTIONS.md)  
**Full application:** [`integration/`](../../integration/)

Local LLM generation with Ollama, evidence-grounded synthesis, query
understanding, and an optional LoRA adapter.

Further documentation:

- [`docs/RUNTIME.md`](docs/RUNTIME.md) — Ollama setup, models, CLI commands
- [`docs/PROMPTS.md`](docs/PROMPTS.md) — prompt families and input contracts
- [`docs/ADAPTER.md`](docs/ADAPTER.md) — LoRA training and deployment

## Model policy

- **Base:** Ollama `qwen3:8b` / Hugging Face `Qwen/Qwen3-8B`
- **Adapter:** QLoRA on the same base, deployed as Ollama `qwen3-research-lora:latest`
- No alternate fallback model families in defaults

## Commands

| Command | Input contract | Retrieval | Main outputs |
| --- | --- | --- | --- |
| `analyze-query` | user query string | never | JSON to stdout or `--out` |
| `chat` | conversational text | never | response Markdown and generation metadata |
| `summarize` | validated `ParsedPaper` JSON | never | summary Markdown and generation metadata |
| `synthesize` | validated `RagEvidencePack` JSON | upstream only | synthesis Markdown, `AnalysisResult`, handoff |
| `compare-prompts` | fixed prompt JSONL | uses fixtures | `results/prompt_comparison/` |
| `compare-models` | fixed prompt JSONL | uses fixtures | `results/model_comparison/` |

All generation commands require a running Ollama daemon and a pulled model tag.

```bash
ollama pull qwen3:8b
python -m app.cli summarize --paper <parsed-paper.json> --out outputs/paper_summary.md
python -m app.cli synthesize --evidence <rag_evidence_pack.json> --out outputs/llm_analysis.md
```

During end-to-end runs, outputs are written to `integration/outputs/`.

## Query understanding

`chat`, `summarize`, and `synthesize` default to `--style auto`. Query analysis
embeds the message with pinned `sentence-transformers/all-MiniLM-L6-v2`, then
reranks low-confidence fields with `cross-encoder/stsb-TinyBERT-L4`.

```bash
python -m app.cli analyze-query --query "I'm confused, but show the mathematics"
```

Integration chat routes conversational text to direct local LLM generation
without calling retrieval.

## Paper summary vs RAG synthesis

Use `summarize` when one parsed paper has already been supplied and the output
must be based only on that paper.

Use `synthesize` when retrieval has already produced a `RagEvidencePack` and the
goal is related-paper or topic-level analysis.

## LoRA dataset and adapter

Full dataset pipeline: [`lora_dataset/README.md`](lora_dataset/README.md).  
Training notebook: [`notebooks/train_lora_adapter.ipynb`](notebooks/train_lora_adapter.ipynb).  
Deployment: [`docs/ADAPTER.md`](docs/ADAPTER.md).

Training data is delivered via `setup_assets.py`. The adapter archive installs
to `models/releases/`; extract to `models/adapters/research_lora_adapter/` before
building the Ollama tag.

## Evaluation artifacts

| Artifact | Path |
| --- | --- |
| Model comparison | `results/model_comparison/` |
| Prompt comparison | `results/prompt_comparison/` |
| Fixed eval prompts | `data/eval/fixed_prompts*.jsonl` |
| Training run record | `results/model_comparison/lora_training_run_20260609.md` |

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Unit tests mock Ollama generation so they run without a local daemon.

## Integration handoff

- Retrieval module provides `RagEvidencePack` via `integration/outputs/rag_evidence_pack.json`.
- For a single uploaded paper, integration calls `summarize` with the PDF parser's
  `ParsedPaper` artifact (no retrieval).
- For topic synthesis, retrieval must run first and provide a real `RagEvidencePack`.
