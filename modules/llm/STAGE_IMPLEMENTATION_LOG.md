# Stage Implementation Log

## Context Read

The implementation follows all Markdown plans under `major_proj`, especially:

- `team_plans/00_project_overview.md`
- `team_plans/01_mulkraj_llm_adapter_prompts.md`
- `team_plans/06_integration_contract.md`
- `team_plans/07_design_rationale_and_defense.md`
- `team_plans/mulkraj_stages/*.md`
- Other member plans for schema, handoff, evidence, and demo expectations.

## Design Decision

No existing project code was present under `major_proj`. The workstream is therefore
implemented as `modules/llm`, with its own `app` package and
stage artifacts. This avoids editing or deleting the existing plan files.

## Offline-First Runtime

Generation uses a local Ollama daemon. Report claims require measured Ollama
outputs from the fixed prompt set and dated `results/model_comparison/ollama_*`
artifact directories.

## Current Execution Paths

- `summarize`: required `ParsedPaper` → query analysis → paper-only prompt → runtime
  → Markdown and generation metadata. Retrieval is explicitly disabled.
- `synthesize`: required `RagEvidencePack` → query analysis → RAG prompt → runtime
  → Markdown, structured `AnalysisResult`, metadata, and handoff.
- `analyze-query`: deterministic structured classification with a replaceable
  `QueryAnalyzer` protocol.
- `chat`: direct local generation for conversational text without retrieval.
- integration `chat`: conversational text uses direct LLM generation; substantive
  questions retain recommender evidence retrieval.

Fixtures are restricted to tests and fixed evaluation prompt generation. Production
commands no longer create sample inputs when a required file is missing.

Standard algorithm maintenance is delegated to installed libraries: arXiv RAG ranking
uses `rank_bm25.BM25Okapi`, and EDA percentiles use pandas nearest quantiles.

## Replacement Path For Final Runs

1. Run the same `data/eval/fixed_prompts.jsonl` through the selected base model.
2. Run the same prompts through quantized variants.
3. Train the LoRA/QLoRA adapter using the documented manifest and config.
4. Record measured outputs under `results/model_comparison/ollama_*`.
5. Regenerate `results/prompt_comparison/` with `compare-prompts` when prompt-strategy tables are needed.
