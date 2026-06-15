# Human Review: Base vs LoRA

Date: 2026-06-14

## Evaluation Basis

The primary comparison contains 12 error-free Ollama generations under
`ollama_20260614/`: six fixed tasks, two model variants, and the same few-shot
strategy. The accepted targeted guard regression remains under
`ollama_20260614_citation_guard_enforced/`; intermediate iteration payloads were
removed after their decisions were captured below.

Human acceptance checks:

- answers the requested task rather than copying input or an example;
- does not claim absent evidence when supplied sections contain it;
- does not infer findings or relevance beyond supplied evidence;
- follows the required output structure;
- uses only supplied citation metadata and source IDs.

Automated structure, source-ID, and citation checks are consistency signals.
They do not establish semantic correctness by themselves.

## Results

| Task | Base `qwen3:8b` | LoRA `qwen3-research-lora:latest` | Decision |
| --- | --- | --- | --- |
| P01 paper summary | Pass after production prompt hardening | Fail: mostly copies the abstract and omits required synthesis | Base |
| P02 topic synthesis | Pass: accurately says retrieved evidence is not directly relevant | Fail: changes the query to robotics | Base |
| P03 research-gap identification | Fail: several absence claims are weak assumptions | Fail: contradicts supplied metrics, baselines, and tasks | Neither |
| P04 citation recommendation | Raw model output fails the strict relevance boundary; enforced guard passes | Raw output fails or copies prompt content; enforced guard passes | Guarded system |
| P05 peer review | Pass: constructive and grounded in sections/structural checks | Fail: adds unsupported claims about missing detail/available sections | Base |
| P06 beginner explanation | Pass: useful method and result explanation | Fail: mostly repeats the abstract | Base |

## Citation Iterations

1. `ollama_20260614_corrected/`: fixed the mixed paper/RAG input contract.
2. `ollama_20260614_citation_hardened/`: added relevance and rejection rules.
3. `ollama_20260614_citation_sanitized/`: removed nested retrieval prompts,
   scores, and ranking reasons from model input.
4. `ollama_20260614_citation_guard_enforced/`: applied a deterministic evidence
   coverage gate after generation.

The final guarded output correctly states that no directly relevant citation is
available. It lists `2102.00002` only as a nearby lead and explicitly records
the missing query concepts: augmented, continuous, data, and visual. Both final
rows have empty errors, source faithfulness 1.0, and citation safety 1.0.

## Model Selection

Keep `qwen3:8b` as the production default. The adapter is not superior on the
fixed tasks and frequently copies source/example text or makes unsupported
absence claims. The June 9 QLoRA run remains valid fine-tuning evidence, but it
does not justify deploying the adapter.

P03 is retained as negative evidence and a known limitation. Do not claim that
the system reliably discovers novel research gaps without stronger structured
evidence and human review.
