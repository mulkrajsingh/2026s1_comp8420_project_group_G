# LLM module

**Module:** LLM — [`docs/CONTRIBUTIONS.md`](../../docs/CONTRIBUTIONS.md)  
**Full app:** [`integration/`](../../integration/)

Staged plan: `team_plans/mulkraj_stages/`.
It is self-contained so the original planning files remain unchanged.

## Model Policy

- **Base:** Ollama `qwen3:8b` / Hugging Face `Qwen/Qwen3-8B`
- **Adapter:** QLoRA on the same base, deployed as Ollama `qwen3-research-lora:latest`
- **No generation fallback models** (no Qwen2.5 or other families in defaults)

## What Is Implemented

- Shared-contract fixtures for `ParsedPaper`, `Recommendation`, and `RagEvidencePack`.
- `python -m app.cli chat` for direct conversational generation without retrieval.
- `python -m app.cli summarize` for summarizing one supplied `ParsedPaper` without retrieval.
- `python -m app.cli synthesize` for evidence-grounded synthesis.
- `python -m app.cli analyze-query` for semantic intent, emotion, expertise,
  verbosity, and response-style classification.
- `python -m app.cli compare-prompts` for zero-shot, few-shot, ReAct, and verification evidence.
- `python -m app.cli compare-models` for base vs adapter comparison artifacts.
- Local Ollama generation with run metadata: backend, model tag, prompt id, latency, error status, and evidence IDs used.
- Assignment-2-style infrastructure: root `requirements.txt`, `training/`, and [`notebooks/`](notebooks/README.md)
  (`model_comparison`, `train_lora_adapter`).

All generation commands require a running Ollama daemon and a pulled model tag.

## Command Reference

| Command | Input contract | Retrieval | Main outputs |
| --- | --- | --- | --- |
| `analyze-query` | user query string | never | JSON to stdout or `--out` |
| `chat` | conversational text | never | response Markdown/text and generation metadata |
| `summarize` | validated `ParsedPaper` JSON | never | summary Markdown and `<stem>_generation.json` |
| `synthesize` | validated `RagEvidencePack` JSON | already completed upstream | synthesis Markdown, `AnalysisResult`, generation metadata, handoff |
| `compare-prompts` | fixed prompt JSONL | uses prompt fixtures | prompt comparison artifacts |
| `compare-models` | fixed prompt JSONL | uses prompt fixtures | model and adapter comparison artifacts |

Production commands require their input files. Sample fixtures are used only by tests
and fixed evaluation prompt generation; missing `ParsedPaper` or `RagEvidencePack`
files are reported as errors.

Standard algorithms use maintained dependencies rather than local reimplementations:

- arXiv RAG retrieval uses `rank_bm25.BM25Okapi` while preserving the module's
  tokenization, evidence schema, score normalization, and no-hit behavior.
- LoRA dataset EDA percentiles use
  `pandas.Series.quantile(interpolation="nearest")`; empty input remains `0`.

## Automatic Query Understanding

`chat`, `summarize`, and `synthesize` default to `--style auto`. Query analysis
first embeds the message and labeled examples with pinned
`sentence-transformers/all-MiniLM-L6-v2`, then compares them with cosine
similarity. Any field below `0.70` confidence is reranked by the pinned
`cross-encoder/stsb-TinyBERT-L4` model. Both models run locally after their first
download/cache population.

```bash
python -m app.cli analyze-query --query "I'm confused, but show the mathematics"
```

Example output:

```json
{
  "intent": "request_explanation",
  "emotion": "confused",
  "topic_expertise": "advanced",
  "verbosity": "detailed",
  "style": "technical",
  "confidence": 0.72,
  "cosine_confidence": 0.34,
  "fallback_used": true
}
```

`--style auto` is the default. Explicit
`--style concise|technical|beginner|reviewer` overrides only the inferred style;
the other query attributes remain semantically classified. Output metadata
includes per-field cosine scores, final confidence, field sources, fallback
fields, and model identifiers.
Expertise is inferred from the current topic query only; it is not stored as a user
profile. Integration chat routes conversational text to direct local LLM generation
without calling retrieval.

## Automatic Thinking Policy

Ollama reasoning is selected from the resolved task and intent:

- Classified direct chit-chat uses `think=false`, an 8,192-token context, and
  at most 512 generated tokens.
- Paper summarization, PDF question answering, peer review, topic/RAG synthesis,
  and other technical requests use `think=true`, the full 40,960-token
  `qwen3:8b` context, and at least 8,192 generated tokens.
- Missing or ambiguous routing metadata fails toward the technical policy.

Generation metadata and production session traces record the selected policy,
context window, and output budget. Hidden reasoning text is never written to
artifacts or session logs.

Set `QUERY_ANALYZER_LOCAL_FILES_ONLY=1` to prohibit model downloads. Optional
`QUERY_EMBEDDING_MODEL`, `QUERY_EMBEDDING_REVISION`, `QUERY_FALLBACK_MODEL`, and
`QUERY_FALLBACK_REVISION` values make model selection explicit; missing configured
models fail instead of reverting to rules.

## Paper Summary vs RAG Synthesis

Use `summarize` when one parsed paper has already been supplied and the output must be
based only on that paper:

```bash
python -m app.cli summarize \
  --paper outputs/parsed_paper.json \
  --query "Explain the method in simple terms" \
  --style auto \
  --model qwen3:8b \
  --out outputs/paper_summary.md
```

This command validates a `ParsedPaper`, does not create or load a `RagEvidencePack`,
does not run retrieval, and writes metadata to
`outputs/paper_summary_generation.json` by default.

Use `synthesize` when retrieval has already produced a `RagEvidencePack` and the goal
is related-paper or topic-level analysis:

```bash
python -m app.cli synthesize \
  --evidence outputs/rag_evidence_pack.json \
  --style auto \
  --model qwen3:8b \
  --out outputs/llm_analysis.md
```

## Unit Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Unit tests mock Ollama generation so they run without a local daemon.

## Ollama Runs

Start Ollama separately and pull/create the model tags you need:

```bash
ollama pull qwen3:8b
```

To build the team adapter tag from the trained PEFT adapter, install/use a local
llama.cpp checkout and run:

```bash
python scripts/build_ollama_research_lora_model.py --check-only

python scripts/build_ollama_research_lora_model.py \
  --llama-cpp /path/to/llama.cpp
```

This creates `qwen3-research-lora:latest` through Ollama using the low-memory adapter
path: `FROM qwen3:8b` plus the converted LoRA adapter GGUF. It does not auto-clone
llama.cpp or fall back to another model family. The full merged F16 GGUF path is
available only with `--merge-mode full`.

Then run:

```bash
python -m app.cli summarize \
  --model qwen3:8b \
  --paper outputs/parsed_paper.json \
  --style technical \
  --out outputs/paper_summary.md

python -m app.cli synthesize \
  --model qwen3:8b \
  --evidence outputs/rag_evidence_pack.json \
  --style technical \
  --out outputs/llm_analysis.md \
  --json-out outputs/analysis_result_from_llm.json

python -m app.cli compare-prompts \
  --model qwen3:8b \
  --test-set data/eval/fixed_prompts.jsonl \
  --out results/prompt_comparison/

python -m app.cli compare-models \
  --models base=qwen3:8b,lora=qwen3-research-lora:latest \
  --test-set data/eval/fixed_prompts.jsonl \
  --out results/model_comparison/
```

Use Ollama rows with empty `error` for model-quality, latency, quantization, and adapter claims.

## LoRA dataset pipeline

Full documentation: [`lora_dataset/README.md`](lora_dataset/README.md).

From `modules/llm/`:

```bash
cp ../.env.example ../.env   # repo root; set KAGGLE_API_TOKEN
pip install kaggle datasets huggingface_hub
python -m lora_dataset.create_dataset
```

Outputs: hybrid open JSONL (~14k rows), Kaggle raw snapshot (~5 GB), random 3k
corpus, 3,000 project RAG rows (500×6 tasks), and a local merged train file
(**16,998** rows with the current default seeds). No synthetic corpus fallback.

The completed 9 June 2026 Colab run used a separately published 17,298-row
training JSONL. Its metrics and provenance are recorded in
`results/model_comparison/lora_training_run_20260609.md`; do not describe the
local 16,998-row manifest as the exact input to that run.

Useful flags: `--skip-download`, `--skip-hybrid`, `--rebuild-corpus`, `--no-seeds`. Optional: `bash scripts/create_dataset.sh`.

## Adapter Workflow

After the pipeline above:

Train on a GPU runtime (Colab or local Jupyter) with `notebooks/train_lora_adapter.ipynb` — QLoRA, step-based training, train/val/test split and eval, checkpoints, zip download, optional Drive backup.

The submission-ready June 9 adapter is retained as
`models/releases/research_lora_adapter_20260609_114209.zip`, with the active
extracted copy under `models/adapters/research_lora_adapter/`. Checksums,
training configuration, and retained metrics are recorded in
`results/model_comparison/lora_training_run_20260609.md`. The original run did
not retain reliable hardware or wall-clock runtime, so those values must not be
invented.

For local deployment, extract the trained adapter zip to
`models/adapters/research_lora_adapter/`, then run
`python scripts/build_ollama_research_lora_model.py --llama-cpp /path/to/llama.cpp`.
The default deployment converts only the adapter and reuses Ollama `qwen3:8b`; the
original adapter zip remains as provenance evidence.

## Stage Coverage

| Stage | Implemented artifact |
| --- | --- |
| 01 runtime and fixed prompts | `data/eval/fixed_prompts.jsonl`, `results/model_comparison/ollama_20260614/runtime_notes.md` |
| 02 evidence-grounded prompt library | `outputs/llm_analysis.md`, `app/prompt_library.py` |
| 03 few-shot vs zero-shot | `compare-prompts` → `results/prompt_comparison/` (Ollama) |
| 04 ReAct tool-call prompting | Live `react_loop.py` on topic retrieval chat (`search_offline`); static examples in `compare-prompts` |
| 05 self-verification | `app/verification.py` same-model pass + `app/faithfulness.py` deterministic audit |
| 06 training data preparation | `python -m lora_dataset.create_dataset`, `data/processed/final_dataset/` |
| 07 adapter training workflow | `notebooks/train_lora_adapter.ipynb`, `results/model_comparison/adapter_training_config.md` |
| 08 base vs LoRA comparison | `results/model_comparison/ollama_20260614/base_vs_lora_table.md`, `model_generations.jsonl`, `human_review.md` |
| 09 local LLM-as-judge | Removed; same-model self-verification in `app/verification.py` instead |
| 10 final synthesis and privacy | `chat`, `summarize`, `synthesize`, generation metadata, no-retrieval disclosure |
| 11 query understanding and routing | `analyze-query`, direct `chat`, retrieval-aware integration routing |

## Integration Notes

Bank should provide `outputs/rag_evidence_pack.json`. Sidharth can call `synthesize`
and render either `outputs/llm_analysis.md` or `outputs/analysis_result_from_llm.json`.

For a single uploaded paper, call `summarize` with the PDF parser's `ParsedPaper`
artifact. This route intentionally bypasses Bank's retrieval output.

For topic synthesis, retrieval must run first and provide a real `RagEvidencePack`.
Integration chat sends classified conversational text to the local LLM without the
recommender; substantive questions retain the existing evidence retrieval path.
