# Report, Presentation, and Video Evidence Index

Last updated: 2026-06-13

Single checklist for the assignment’s **22 + 11 + 5 + 2** allocation. Claim
readiness is authoritative only when the referenced artifact exists and its
backend/source labels support the wording.

## Submission Artifacts

| Deliverable | Allocation | Status | Path |
| --- | ---: | --- | --- |
| Project code/techniques/integration | 22 | mixed/strong | Canonical `modules/` and `integration/` paths |
| Report PDF | 11 | missing | `Group_G_Assignment3/Report/Group_G_Report.pdf` |
| Presentation PDF | 5 | present | `Group_G_Assignment3/Presentation/Presentation.pdf` |
| Video <=5 minutes | 2 | missing | `Group_G_Assignment3/Video/` |
| Submission code tree | required | missing | `Group_G_Assignment3/Codes/` |

## Basic Techniques

| Technique | Readiness | Primary evidence |
| --- | --- | --- |
| Corpus preprocessing/validation | ready | `modules/dataset/data/processed/dev_5k.jsonl`, `results/data_validation/paperrecord_validation.json` |
| Semantic Scholar enrichment | ready | `dev_5k_enriched.jsonl`, `s2_enrichment_summary.{json,md}`: 4,797/5,000 matched |
| Dataset EDA | ready | `modules/dataset/results/eda/figures/`, subset policy and limitations |
| TF-IDF/BM25 | ready | `modules/retrieval/app/retrieval/tfidf_bm25.py`, retrieval metrics |
| Rule-based PDF parsing | ready | `modules/pdf_nlp/pdf_parser.py`, parser tests, 52-reference fixture |
| POS/NER/keyphrases | ready, limited corpus | `modules/pdf_nlp/results/pdf_nlp/`; provisional five-paper labels |
| Traditional classifier | ready, internal split | LogReg/SVM metrics and confusion matrix under `modules/dataset/results/classification/` |
| Extractive summarization | ready, limited corpus | TextRank source traceability 1.000 on five papers |

## Advanced Techniques

| Technique | Readiness | Primary evidence and qualification |
| --- | --- | --- |
| RAG and APA recommendations | ready | `modules/retrieval/app/rag_pack.py`, `app/citation.py`, curated `integration/results/demo/pdf_ollama_related/analysis_result.json` |
| 5,000-paper retrieval evaluation | mixed | `results/retrieval/retrieval_comparison.csv`; only five gold queries |
| Hybrid retrieval | mixed | Implemented/tuned, but TF-IDF leads aggregate metrics and two metadata signals are inactive |
| Prompt engineering/ReAct | ready for implementation claim | `modules/llm/app/prompt_library.py`, `compare-prompts`; regenerate `results/prompt_comparison/` with Ollama for measured rows |
| Local Qwen3 runtime | ready | Ollama-backed production path in `integration/` and `modules/llm/` |
| Team QLoRA training | ready for training claim | June 9 notebook run, release archive, hashes, train/validation/test metrics |
| Base-vs-LoRA quality | ready with negative result | `ollama_20260614/` plus human review; base selected |
| Faithfulness/LLM-as-judge | ready with limits | Automated artifacts plus committed human review |
| BGE-M3 | not implemented | Mention only as considered/rejected |

## Integration Evidence

| Item | Readiness | Path |
| --- | --- | --- |
| Shared pipeline/contracts | ready | `integration/app/pipeline.py`, `team_plans/06_integration_contract.md` |
| CLI and FastAPI | ready | `integration/app/cli.py`, `integration/app/api.py` |
| Canonical Vite UI | ready locally | Build and single-request chat browser run; screenshots under `integration/results/demo/` |
| Topic live/Ollama trace | ready with source labels | `integration/results/traces/topic_ollama_20260605.jsonl` |
| PDF paper-only Ollama trace | ready | `pdf_drq_ollama_paper_only_20260613.jsonl` |
| PDF related-paper Ollama trace | ready | `pdf_drq_ollama_related_20260613.jsonl` |
| UI topic Ollama trace | ready | `topic_ollama_ui_20260613.jsonl` |
| Runtime outputs/sessions | generated, not evidence by default | ignored `integration/outputs/`, `integration/data/sessions/` |

## Adapter Evidence

- Release:
  `modules/llm/models/releases/research_lora_adapter_20260609_114209.zip`
- Active extracted adapter:
  `modules/llm/models/adapters/research_lora_adapter/`
- Run summary:
  `modules/llm/results/model_comparison/lora_training_run_20260609.md`
- Machine-readable metrics:
  `modules/llm/results/model_comparison/lora_training_metrics_20260609.json`
- Canonical notebook:
  `modules/llm/notebooks/train_lora_adapter.ipynb`

The completed run used 17,298 rows. The local reproducible pipeline’s current
manifest has 16,998 rows; do not conflate the two datasets.

## Report-Ready Visuals

- Dataset: `modules/dataset/results/eda/figures/*.png`
- Retrieval:
  `modules/retrieval/results/retrieval/retrieval_comparison_chart.png`,
  `ndcg_heatmap.png`, `hybrid_ranking_breakdown.png`,
  `hybrid_weight_sensitivity.png`
- Model tables:
  `modules/llm/results/model_comparison/ollama_20260614/base_vs_lora_table.md`,
  `ollama_20260614/final_model_comparison.csv`
- Architecture: `docs/ARCHITECTURE.md`
- PDF-NLP metrics: `modules/pdf_nlp/results/pdf_nlp/comparison_report.md`
- Frontend single-request chat:
  `integration/results/demo/frontend_single_request_chat.png`

## Do Not Claim

- Model-quality gains without Ollama-backed traces
- Universal LoRA superiority
- Robust general retrieval performance from only five queries
- Hybrid retrieval is best overall in the committed results
- BGE-M3 is implemented
- SciER or hybrid NER is superior to the deterministic baseline
- Nadiyah's five-paper provisional evaluation generalises to other papers
- Raw arXiv metadata contains full paper bodies
- Zero hallucination
- Missing hardware/runtime values for the June 9 training run

## Final Submission Checklist

- [ ] Report PDF under `Group_G_Assignment3/Report/`
- [ ] Code/notebooks under `Group_G_Assignment3/Codes/`
- [x] Presentation PDF under `Group_G_Assignment3/Presentation/`
- [ ] Video under `Group_G_Assignment3/Video/`
- [ ] Yash classifier metrics
- [x] Nadiyah POS/NER/keyphrase, extractive-summary, and structural-check evidence
- [ ] Final error-free Ollama base-vs-adapter comparison with human review
- [x] Final Vite build/browser screenshot
- [ ] <=5-minute demo recording
