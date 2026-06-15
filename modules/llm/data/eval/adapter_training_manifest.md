# Adapter Training Manifest

Purpose: train a small academic-style LoRA/QLoRA adapter that improves structure,
citation safety, tool-call JSON reliability, and peer-review formatting. It should
not memorize the arXiv corpus.

| Source | Planned task | Use |
| --- | --- | --- |
| SciTLDR | paper-to-summary | concise scientific summaries |
| QASPER | evidence-to-answer | grounded QA over paper text |
| SciCite | citation intent | citation recommendation language |
| PeerRead-style data | paper-to-review | reviewer-style critique |
| Local fixed prompts | structure/tool-call examples | project-specific output format |

Local fixed prompt records available: 6.

Required filtering: keep only open/allowed data, store provenance, remove examples
with missing source text, and keep train/eval splits deterministic.
