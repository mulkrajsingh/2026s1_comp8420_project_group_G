# Production PDF-NLP Real-Paper Evaluation

**Corpus:** five real research-paper PDFs; no mock paper data.

**Annotations:** provisional agent-reviewed labels, not human-validated ground truth.

Results apply only to this local five-paper corpus and must not be generalised.

## Aggregate Metrics

| Metric | Result |
| --- | ---: |
| Successful papers | 5/5 |
| Parser title similarity | 0.789 |
| Parser section recall | 0.804 |
| POS accuracy | 0.900 |
| SciER NER F1 | 0.226 |
| Baseline NER F1 | 0.621 |
| Hybrid NER F1 | 0.280 |
| Keyphrase Precision@10 | 0.380 |
| Keyphrase concept recall | 0.260 |
| Extractive summary concept retention | 0.400 |
| Extractive summary source traceability | 1.000 |
| Structural checklist F1 | 0.453 |

## Provenance

- SciER track: Nadiyah's fine-tuned DistilBERT at threshold 0.7.
- Baseline track: deterministic gazetteer, metric regex, and spaCy organisations.
- Hybrid track: union of SciER and baseline mentions.
- Keyphrases: Nadiyah's KeyBERT approach with local MiniLM.
- Summary: deterministic extractive TextRank; BART remains historical comparison only.

Runtime failures: 0. See `failure_cases.md`.
