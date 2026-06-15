# LLM module handoff

## What This Module Produces

- `outputs/llm_analysis.md`: evidence-grounded synthesis from a `RagEvidencePack`.
- `outputs/analysis_result_from_llm.json`: `AnalysisResult`-compatible JSON for integration CLI/API.
- `outputs/llm_generation.json`: generation metadata for the latest synthesis run.
- `data/eval/fixed_prompts.jsonl`: six fixed prompts for repeatable comparison.
- `results/prompt_comparison/`: Ollama-backed prompt strategy evidence from `compare-prompts`.
- `results/model_comparison/`: model, adapter, quantization, and judge evidence.

## How Integration Should Consume It

1. `modules/retrieval` writes `outputs/rag_evidence_pack.json`.
2. Run `python -m app.cli synthesize --evidence outputs/rag_evidence_pack.json --style technical --out outputs/llm_analysis.md`.
3. For model comparison evidence, pass `--model <tag>` when running `compare-models` or `compare-prompts`.
4. Render `outputs/llm_analysis.md` in CLI/UI, or consume `outputs/analysis_result_from_llm.json`.

## Current Limitation

Report claims about output quality, latency, quantization, or adapter improvement
must use Ollama/model-comparison artifacts created from real local model tags.
