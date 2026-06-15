# Nadiyah vs Reference Pipeline: Real-PDF Evaluation

**Corpus:** 5 real research-paper PDFs. No mock paper data.
**Runtime mode:** forced offline.
**Annotations:** provisional agent-reviewed labels; not human-validated ground truth.

## Executive Result

| Track | Planned features implemented | Claim status |
| --- | ---: | --- |
| Nadiyah as supplied | 0/6 | pending |
| Nadiyah minimally repaired | 4/6 | mixed |
| Independent reference pipeline | 6/6 | implemented as comparison only |

The original submission cannot be performance-tested end to end because imports, preprocessing, parser CLI execution, and real-abstract NER inference fail. The repaired track measures the same submitted algorithms after compatibility fixes; the reference track is independent and must not be attributed to Nadiyah.

## Execution

| Track | Successful papers | Mean runtime/paper | Mean throughput | Observed process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Nadiyah as supplied | 0/5 | n/a | n/a | n/a |
| Nadiyah minimally repaired | 5/5 | 17.636 s | 1.152 pages/s | 2485.0 MB |
| Independent reference | 5/5 | 0.432 s | 65.310 pages/s | 2485.0 MB |

Throughput is PDF pages divided by full per-paper pipeline runtime. Peak RSS is the cumulative high-water mark of the shared evaluation process, so it is not an isolated component-memory benchmark.

## Aggregate Comparison

| Metric | Nadiyah repaired | Reference |
| --- | ---: | ---: |
| Parser title similarity | 0.200 | 1.000 |
| Parser section recall | 0.000 | 1.000 |
| Parser reference-count absolute error | 50.800 | 9.600 |
| POS accuracy (50 tokens) | 0.820 | 0.920 |
| NER overlap F1 | 0.637 | 0.942 |
| Keyphrase Precision@10 | 0.660 | 0.520 |
| Keyphrase concept recall | 0.540 | 0.500 |
| Summary concept retention | 0.280 | 0.500 |
| Summary source traceability | 0.666 | 1.000 |
| Structural checklist F1 | unimplemented | 1.000 |

## NER Threshold Analysis

| Nadiyah threshold | Exact F1 | Overlap F1 |
| ---: | ---: | ---: |
| 0.5 | 0.500 | 0.635 |
| 0.7 | 0.504 | 0.637 |
| 0.9 | 0.475 | 0.593 |

## Interpretation

- Nadiyah repaired performs better on: keyphrase Precision@10 (0.660 vs 0.520), keyphrase concept recall (0.540 vs 0.500)
- Reference performs better on: parser section recall (1.000 vs 0.000), POS accuracy (0.920 vs 0.820), summary concept retention (0.500 vs 0.280), summary traceability (1.000 vs 0.666)
- Nadiyah repaired NER overlap F1 at threshold 0.7: 0.637.
- Reference NER overlap F1: 0.942.
- Nadiyah's BART summaries are abstractive. They are useful supporting evidence only and do not close the extractive-summary gap.
- The reference pipeline supplies deterministic structural checks and TextRank summaries, but it is comparison code rather than production integration.

## Historical SciER Artifact

- Highest recorded validation F1: 0.771 at epoch 6.0.
- Recorded selected checkpoint: `./scier-distilbert\checkpoint-1745`; present: `False`.
- Final model labels: `{'0': 'O', '1': 'Method', '2': 'Dataset', '3': 'Task'}`.
- Retained checkpoint labels: `{'0': 'LABEL_0', '1': 'LABEL_1', '2': 'LABEL_2', '3': 'LABEL_3', '4': 'LABEL_4', '5': 'LABEL_5', '6': 'LABEL_6'}`.
- No saved final test-metric artifact was supplied. Historical validation metrics are not used as real-PDF comparison scores.

## Failures And Limitations

- Recorded component/paper failures: 0. See `failure_cases.md`.
- The gold set is small and provisionally agent-reviewed.
- Gazetteer-based reference NER is intentionally interpretable and tailored to scientific ML terminology; it is not a broad benchmark.
- Process peak RSS is cumulative within one evaluation process, not isolated per component.
- Parser reference counts use bibliography-year detection and should be interpreted as approximate.

## Defensible Claim Status

- **Nadiyah original:** pending; source artifacts exist but the supplied workflow is not executable.
- **Nadiyah repaired:** mixed; POS, SciER NER, KeyBERT, BART, and experimental parsing execute locally, but parser quality is weak and planned extractive/structural features are absent.
- **Reference:** implemented for comparison evidence only; not production-integrated and not attributable to Nadiyah.

## Offline Repeatability

- Semantic outputs matching: 10/10.
- All functional semantic outputs match: `True`.
- Timing, process memory, full text, and the non-functional markdown character-count diagnostic are excluded from semantic hashes.
- Non-functional parser diagnostic variation occurred in 2/10 outputs; retained sections and scored NLP outputs were unchanged.
