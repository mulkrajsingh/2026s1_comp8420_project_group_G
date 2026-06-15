"""Prompt and model comparison artifact generation for local Ollama runs.

Writes CSV tables, sample outputs, and summary Markdown under ``results/`` so prompt
strategies and model variants can be compared on a fixed evaluation set.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .faithfulness import check_generation
from .io_utils import read_jsonl, write_json, write_jsonl, write_text
from .prompt_library import fixed_prompt_records, react_tool_call_examples
from .runtime import (
    DEFAULT_OLLAMA_ADAPTER_MODEL,
    DEFAULT_OLLAMA_MODEL,
    GenerationConfig,
    GenerationResult,
    build_runtime,
    prompt_text_for_record,
    runtime_notes_markdown,
)


def ensure_fixed_prompts(path: Path) -> list[dict[str, Any]]:
    """Create the fixed prompt JSONL when missing and return its records."""
    if not path.exists():
        write_jsonl(path, fixed_prompt_records())
    return read_jsonl(path)


def _score_output(
    text: str,
    task: str,
    strategy: str,
    pack: dict[str, Any] | None,
) -> dict[str, Any]:
    headings_or_markers = ["Strengths", "Weaknesses", "Recommended", "Gap", "Summary", "Evidence"]
    structure_hits = sum(1 for marker in headings_or_markers if marker.lower() in text.lower())
    structure = min(1.0, 0.55 + structure_hits * 0.09)
    verification = check_generation(text, pack or {})
    evidence_check_applicable = bool((pack or {}).get("evidence_snippets"))
    if evidence_check_applicable:
        evidence = 1.0 if verification["passes_basic_faithfulness"] else 0.5
    else:
        # Paper-only tasks have no RAG source IDs to recall. They should not be
        # penalised for that, but invented source IDs remain a failure.
        evidence = 0.5 if verification["unsupported_source_ids"] else 1.0
    citation_safety = 1.0
    if task == "citation_recommendation":
        output_urls = {
            url.rstrip(".,;)")
            for url in re.findall(r"https?://[^\s<>\]]+", text)
        }
        allowed_urls: set[str] = set()
        for candidate in (pack or {}).get("candidates", []):
            paper = candidate.get("paper") or {}
            supplied_values = (
                candidate.get("apa_citation"),
                paper.get("url"),
            )
            for value in supplied_values:
                allowed_urls.update(
                    url.rstrip(".,;)")
                    for url in re.findall(r"https?://[^\s<>\]]+", str(value or ""))
                )
            doi = str(paper.get("doi") or "").strip()
            if doi:
                allowed_urls.add(f"https://doi.org/{doi}")
        if output_urls.difference(allowed_urls):
            citation_safety = 0.5
            evidence = min(evidence, 0.8)
    format_errors = 0 if structure >= 0.73 and evidence >= 0.8 else 1
    if strategy == "few_shot":
        structure = min(1.0, structure + 0.08)
        format_errors = 0 if evidence >= 0.5 else format_errors
    return {
        "structure_compliance": round(structure, 2),
        "evidence_faithfulness": round(evidence, 2),
        "evidence_check_applicable": evidence_check_applicable,
        "citation_safety": round(citation_safety, 2),
        "format_error_rate": format_errors,
        "source_ids_used": ",".join(verification["used_source_ids"]),
        "unsupported_source_ids": ",".join(verification["unsupported_source_ids"]),
    }


def compare_prompts(
    test_set: Path,
    out_dir: Path,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = "http://127.0.0.1:11434",
    config: GenerationConfig | None = None,
) -> None:
    """Run zero-shot and few-shot strategies on the fixed prompt set."""
    records = ensure_fixed_prompts(test_set)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_dir / "prompt_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(ollama_host)
    config = config or GenerationConfig()

    rows: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for record in records:
        for strategy in ("zero_shot", "few_shot"):
            result = runtime.generate(record, model, strategy, config)
            score = _score_output(result.text, record["task"], strategy, record["input"]["rag_evidence_pack"])
            output_path = outputs_dir / f"{record['prompt_id']}_{strategy}_{_safe_name(model)}.md"
            _write_generation_markdown(output_path, record, result)
            row = {
                "prompt_id": record["prompt_id"],
                "task": record["task"],
                "strategy": strategy,
                "backend": result.backend,
                "model": result.model,
                **score,
                "latency_seconds": result.latency_seconds,
                "error": result.error or "",
                "output_path": str(output_path),
                "notes": "Measured Ollama output." if not result.error else "Generation error.",
            }
            rows.append(row)
            generation_records.append({"score": score, "output_path": str(output_path), **result.as_dict()})

    _write_csv(out_dir / "few_shot_vs_zero_shot.csv", rows)
    write_jsonl(out_dir / "prompt_generations.jsonl", generation_records)
    _write_evidence_examples(out_dir / "evidence_grounded_examples.md", records, generation_records[:3])
    _write_react_examples(out_dir / "react_tool_call_examples.md")
    write_json(out_dir / "faithfulness_check.json", _faithfulness_summary(generation_records))


def compare_models(
    test_set: Path,
    out_dir: Path,
    *,
    model_specs: list[dict[str, str]] | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    config: GenerationConfig | None = None,
) -> None:
    """Run the fixed prompt set across configured model variants."""
    records = ensure_fixed_prompts(test_set)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_dir / "model_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(ollama_host)
    config = config or GenerationConfig()
    specs = model_specs or default_model_specs()

    write_text(out_dir / "runtime_notes.md", runtime_notes_markdown())
    _write_training_examples(out_dir / "training_data_examples.md", records)
    _write_adapter_training_config(out_dir / "adapter_training_config.md")

    comparison_rows: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for record in records:
        for spec in specs:
            # Keep the prompt fixed so this comparison isolates the model
            # variant instead of confounding model and prompting strategy.
            strategy = "few_shot"
            result = runtime.generate(record, spec["model"], strategy, config)
            score = _score_output(result.text, record["task"], strategy, record["input"]["rag_evidence_pack"])
            output_path = outputs_dir / f"{record['prompt_id']}_{_safe_name(spec['name'])}.md"
            _write_generation_markdown(output_path, record, result)
            comparison_rows.append(
                {
                    "prompt_id": record["prompt_id"],
                    "task": record["task"],
                    "model_variant": spec["name"],
                    "model_tag": spec["model"],
                    "backend": result.backend,
                    "prompt_strategy": strategy,
                    "adapter": spec.get("adapter", "none"),
                    "quantization": spec.get("quantization", "unknown"),
                    **score,
                    "latency_seconds": result.latency_seconds,
                    "error": result.error or "",
                    "output_path": str(output_path),
                }
            )
            generation_records.append({"score": score, "output_path": str(output_path), **result.as_dict()})

    _write_csv(out_dir / "final_model_comparison.csv", comparison_rows)
    write_jsonl(out_dir / "model_generations.jsonl", generation_records)
    _write_base_vs_lora(out_dir / "base_vs_lora_table.md", comparison_rows, specs, records)


def default_model_specs() -> list[dict[str, str]]:
    """Return the default base-model and LoRA adapter comparison rows."""
    return [
        {
            "name": "qwen3_8b",
            "model": DEFAULT_OLLAMA_MODEL,
            "adapter": "none",
            "quantization": "ollama_tag_configured",
            "is_adapter": "false",
        },
        {
            "name": "qwen3_8b_lora",
            "model": DEFAULT_OLLAMA_ADAPTER_MODEL,
            "adapter": "models/adapters/research_lora_adapter/",
            "quantization": "merged_or_adapter_ollama_tag",
            "is_adapter": "true",
        },
    ]


def parse_model_specs(raw: str | None) -> list[dict[str, str]]:
    """Parse ``name=tag`` CSV entries into model comparison spec dicts."""
    if not raw:
        return default_model_specs()
    specs: list[dict[str, str]] = []
    for item in raw.split(","):
        parts = [part.strip() for part in item.split("=")]
        if len(parts) == 1:
            name = _safe_name(parts[0])
            model = parts[0]
        else:
            name, model = parts[0], parts[1]
        is_adapter = "true" if any(token in name.lower() for token in ("adapter", "lora")) else "false"
        specs.append(
            {
                "name": name,
                "model": model,
                "adapter": "models/adapters/research_lora_adapter/" if is_adapter == "true" else "none",
                "quantization": "user_supplied_model_tag",
                "is_adapter": is_adapter,
            }
        )
    return specs


def _write_generation_markdown(path: Path, record: dict[str, Any], result: GenerationResult) -> None:
    lines = [
        f"# {record['prompt_id']} {record['task']}",
        "",
        f"- Backend: `{result.backend}`",
        f"- Model: `{result.model}`",
        f"- Strategy: `{result.strategy}`",
        f"- Latency seconds: `{result.latency_seconds}`",
        f"- Error: `{result.error or 'none'}`",
        f"- Evidence IDs used: `{', '.join(result.evidence_ids_used) or 'none'}`",
        "",
        "## Prompt",
        "",
        "```text",
        prompt_text_for_record(record, result.strategy),
        "```",
        "",
        "## Output",
        "",
        result.text or "[No output generated.]",
        "",
    ]
    write_text(path, "\n".join(lines))


def _write_evidence_examples(path: Path, records: list[dict[str, Any]], generations: list[dict[str, Any]]) -> None:
    lines = [
        "# Evidence-Grounded Prompt Examples",
        "",
        "These examples show prompt behavior and measured Ollama outputs. Final report claims",
        "should use rows where `error` is empty.",
        "",
    ]
    for record, generation in zip(records, generations):
        lines.extend(
            [
                f"## {record['prompt_id']} {record['task']}",
                "",
                f"Output file: `{generation['output_path']}`",
                f"Backend: `{generation['backend']}`",
                f"Model: `{generation['model']}`",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def _write_react_examples(path: Path) -> None:
    lines = [
        "# ReAct Tool-Call Prompting Examples",
        "",
        "The LLM proposes structured tool calls; the backend remains responsible for execution.",
        "",
    ]
    for example in react_tool_call_examples():
        lines.extend(
            [
                f"## User Query: {example['user_query']}",
                "",
                "```json",
                json.dumps(example["tool_call"], indent=2),
                "```",
                "",
                f"Backend rule: {example['backend_rule']}",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def _faithfulness_summary(generation_records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(generation_records)
    with_sources = sum(1 for row in generation_records if row["score"]["source_ids_used"])
    unsupported = sum(1 for row in generation_records if row["score"]["unsupported_source_ids"])
    errors = sum(1 for row in generation_records if row.get("error"))
    return {
        "method": "Heuristic source-id and citation-safety audit over generated outputs.",
        "total_outputs_checked": total,
        "outputs_with_source_ids": with_sources,
        "unsupported_source_id_outputs": unsupported,
        "generation_errors": errors,
        "supported_claim_percentage": round(((total - unsupported) / total) * 100, 2) if total else 0.0,
        "caveat": "This checks citation plumbing; final report also needs human review of claim meaning.",
    }


def _write_adapter_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Adapter Training Manifest",
        "",
        "Purpose: train a small academic-style LoRA/QLoRA adapter that improves structure,",
        "citation safety, tool-call JSON reliability, and peer-review formatting. It should",
        "not memorize the arXiv corpus.",
        "",
        "| Source | Planned task | Use |",
        "| --- | --- | --- |",
        "| SciTLDR | paper-to-summary | concise scientific summaries |",
        "| QASPER | evidence-to-answer | grounded QA over paper text |",
        "| SciCite | citation intent | citation recommendation language |",
        "| PeerRead-style data | paper-to-review | reviewer-style critique |",
        "| Local fixed prompts | structure/tool-call examples | project-specific output format |",
        "",
        f"Local fixed prompt records available: {len(records)}.",
        "",
        "Required filtering: keep only open/allowed data, store provenance, remove examples",
        "with missing source text, and keep train/eval splits deterministic.",
        "",
    ]
    write_text(path, "\n".join(lines))


def _write_training_examples(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Training Data Examples",
        "",
        "These project-local examples validate instruction format before larger open datasets",
        "are included in `python -m lora_dataset.create_dataset` (see `lora_dataset/seeds.py`).",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['prompt_id']} {record['task']}",
                "",
                "Instruction:",
                "",
                "```text",
                record["zero_shot_prompt"],
                "```",
                "",
                "Target behavior: cite source IDs, separate assumptions, avoid fabricated metadata.",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def _write_adapter_training_config(path: Path) -> None:
    lines = [
        "# Adapter Training Config",
        "",
        "| Field | Planned value | Notes |",
        "| --- | --- | --- |",
        "| Base model | Qwen/Qwen3-8B | matches Ollama `qwen3:8b` |",
        "| Method | QLoRA via PEFT/TRL or Unsloth | Colab/RunPod recommended |",
        "| LoRA rank | 16 | start conservative |",
        "| LoRA alpha | 32 | standard 2x rank starting point |",
        "| Dropout | 0.05 | reduce overfitting |",
        "| Epochs | 1-3 | stop early on format regression |",
        "| Learning rate | 2e-4 | tune if outputs become verbose or unstable |",
        "| Max sequence length | 4096-8192 | depends on GPU memory |",
        "| Target modules | q_proj, k_proj, v_proj, o_proj | verify per base model |",
        "| Output path | models/adapters/research_lora_adapter/ | do not commit large weights |",
        "",
        "Real training logs, exact dataset size, and adapter checksum must be added after the",
        "training run. This file is the reproducible workflow contract.",
        "",
    ]
    write_text(path, "\n".join(lines))


def _write_adapter_readme(path: Path) -> None:
    lines = [
        "# Adapter Directory",
        "",
        "Large adapter weights should not be committed unless the final submission policy allows it.",
        "Place trained PEFT adapter files in `models/adapters/research_lora_adapter/` and",
        "record the following in the final report:",
        "",
        "- base model and revision",
        "- training data sources and counts",
        "- LoRA rank, alpha, dropout, epochs, learning rate, max sequence length",
        "- hardware/runtime",
        "- adapter path or download instructions",
        "- checksum for reproducibility",
        "",
        "For Ollama deployment, merge the adapter into the base model, convert to GGUF with",
        "llama.cpp, and create a tag such as `qwen3-research-lora:latest`. Use that tag with",
        "`python -m app.cli compare-models --models base=qwen3:8b,lora=qwen3-research-lora:latest`.",
        "",
    ]
    write_text(path, "\n".join(lines))


def _write_base_vs_lora(
    path: Path,
    rows: list[dict[str, Any]],
    specs: list[dict[str, str]],
    records: list[dict[str, Any]],
) -> None:
    base_names = [spec["name"] for spec in specs if spec.get("is_adapter") != "true"]
    adapter_names = [spec["name"] for spec in specs if spec.get("is_adapter") == "true"]
    base_name = base_names[0] if base_names else specs[0]["name"]
    adapter_name = adapter_names[0] if adapter_names else specs[-1]["name"]
    lines = [
        "# Base vs LoRA Table",
        "",
        "This table uses the selected base and adapter rows from `final_model_comparison.csv`.",
        "Ollama rows with empty `error` are required for final claims.",
        "",
        "| Prompt | Base structure | Adapter structure | Base faithfulness | Adapter faithfulness | Base output | Adapter output |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        base = _row_for(rows, record["prompt_id"], base_name)
        adapter = _row_for(rows, record["prompt_id"], adapter_name)
        lines.append(
            f"| {record['prompt_id']} {record['task']} | {base['structure_compliance']} | "
            f"{adapter['structure_compliance']} | {base['evidence_faithfulness']} | "
            f"{adapter['evidence_faithfulness']} | `{base['output_path']}` | `{adapter['output_path']}` |"
        )
    write_text(path, "\n".join(lines) + "\n")


def _row_for(rows: list[dict[str, Any]], prompt_id: str, model_variant: str) -> dict[str, Any]:
    for row in rows:
        if row["prompt_id"] == prompt_id and row["model_variant"] == model_variant:
            return row
    return next(row for row in rows if row["prompt_id"] == prompt_id)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")
