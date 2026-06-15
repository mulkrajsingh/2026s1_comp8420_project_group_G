# Canonical Model Comparison Evidence

Two dated result sets are retained:

- `ollama_20260614/`: the fair six-task, 12-generation base-versus-LoRA
  comparison, including raw generations, per-prompt outputs, metrics, runtime
  notes, and human review.
- `ollama_20260614_citation_guard_enforced/`: the accepted targeted citation
  safety regression after the deterministic eligibility guard.

The intermediate corrected, hardened, sanitized, and pre-enforcement citation
runs were removed after their decisions were captured in
`ollama_20260614/human_review.md`.

Top-level training metrics and run notes describe adapter-training evidence.
Prior proxy `prompt_comparison/` and top-level proxy comparison outputs were
removed; use the dated `ollama_*` directories for measured model-quality rows.
