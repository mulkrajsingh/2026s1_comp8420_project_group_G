# PDF-NLP Production Module

**Module:** PDF-NLP — see [`docs/CONTRIBUTIONS.md`](../../docs/CONTRIBUTIONS.md)

**Canonical app:** [`integration/`](../../integration/)
**Model manifest:** [`models/manifest.json`](models/manifest.json)

## Architecture

The production path keeps the existing `pypdf` parser and enriches its
`ParsedPaper` output. It does not use the weaker PyMuPDF4LLM comparison parser.

```text
real PDF
  -> pdf_parser.py: metadata, sections, references
  -> paper_analysis.py:
       spaCy POS/dependencies/noun chunks
       Nadiyah SciER DistilBERT NER (threshold 0.7)
       metric regex + gazetteer + spaCy organisations
       Nadiyah KeyBERT + local MiniLM
       deterministic TextRank extractive summary
       deterministic structural checks
  -> enriched ParsedPaper + analysis/review sidecars
```

All non-empty core sections are processed. The five required section keys
remain present, while `related_work`, `limitations`, and `discussion` are
retained when detected.

## Model Setup

Runtime assets are deliberately ignored under `models/runtime/`. The project
owner supplies the team Google Drive ZIP separately; installation also accepts
any local archive path.

```bash
cd modules/pdf_nlp
python -m app.cli model-assets --archive /absolute/path/to/pdf_nlp_models.zip
python -m app.cli model-assets
```

The installer rejects path traversal, extracts to a staging directory, and
validates required files against tracked SHA-256 values before replacing the
runtime directory. Missing or invalid assets stop production inference; there
is no silent model substitution.

The archive contains only:

- Nadiyah's final SciER DistilBERT model and tokenizer;
- `sentence-transformers/all-MiniLM-L6-v2` for KeyBERT;
- spaCy `en_core_web_sm`.

Training checkpoints, caches, virtual environments, BART weights, and the
historical 769 MB checkpoint set are excluded.

## Commands

Download test PDFs first — see [`tests/README.md`](../../tests/README.md).
Run from `modules/pdf_nlp/`:

```bash
# Lightweight parser only
python -m app.cli parse-pdf \
  --pdf ../../tests/papers/drq_v2/2107.09645v1.pdf \
  --out outputs/parsed_paper.json \
  --debug-out outputs/pdf_parse_debug.json

# Enrich an existing ParsedPaper
python -m app.cli basic-nlp \
  --paper-json outputs/parsed_paper.json \
  --out outputs/basic_nlp.json \
  --enriched-paper-out outputs/enriched_paper.json

# Deterministic structural checks only
python -m app.cli peer-review-checks \
  --paper-json outputs/parsed_paper.json \
  --out outputs/structural_review.json

# Parse, enrich, and review one real paper
python -m app.cli analyze-paper \
  --pdf ../../tests/papers/drq_v2/2107.09645v1.pdf

# Five-paper production evaluation
python -m app.cli evaluate-real-papers \
  --manifest data/evaluation/real_papers.json \
  --annotations data/evaluation/annotations.json \
  --out results/pdf_nlp
```

Evaluation annotations are imported only by `evaluation.py`; production code
cannot read them.

## Output Contract

`ParsedPaper.keywords` and canonical entity groups are populated. The optional
`analysis` object contains:

- compact POS counts, selected token rows, dependencies, lemmas, and noun chunks;
- scored entity mentions with source, section, and offsets;
- per-section keyphrases;
- source-traceable extractive-summary sentences;
- structural findings;
- phase timings and model provenance.

The integrated `AnalysisResult.paper_analysis` exposes this deterministic
evidence without moving PDF parsing or retrieval into `modules/llm`.

## Real-Paper Evaluation

The manifest contains only five repository PDFs: DrQ-v2, SIGA, Transformer,
BERT, and RAG. Their URLs, sizes, page counts, and SHA-256 values are tracked.
Annotations are provisional and not human-validated, so results apply only to
this corpus.

| Metric | Five-paper result |
| --- | ---: |
| Successful papers | 5/5 |
| Parser title similarity | 0.789 |
| Parser section recall | 0.804 |
| POS accuracy | 0.900 |
| SciER NER F1 | 0.226 |
| Deterministic baseline NER F1 | 0.621 |
| Hybrid NER F1 | 0.280 |
| Keyphrase Precision@10 | 0.380 |
| Extractive-summary traceability | 1.000 |
| Structural checklist F1 | 0.453 |

The baseline outperforming SciER and hybrid is a measured limitation, not a
reason to hide those rows. See [`results/pdf_nlp/`](results/pdf_nlp/).

## Provenance

- Nadiyah: SciER fine-tuning, KeyBERT approach, and repaired comparison work.
- Independent integration additions: canonical parser preservation, spaCy
  syntax/organisation extraction, deterministic TextRank, metric rules,
  structural checks, model installer, production CLI, and integration wiring.
- BART is retained only in
  [`results/historical_comparison/`](results/historical_comparison/)
  and is not a production dependency.

## Verification

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
SKIP_OLLAMA=1 ../../tests/run_system_tests.sh
```

Current verified module result: 23 PDF-NLP tests pass. DrQ-v2 produces 52
references, 3,963 POS tokens, 20 keyphrases, five extractive sentences, and
source-labelled entity mentions.
