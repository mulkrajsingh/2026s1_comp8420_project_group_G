# Dataset module

**Module:** Dataset — see [`docs/CONTRIBUTIONS.md`](../../docs/CONTRIBUTIONS.md)

Corpus export, Semantic Scholar enrichment, validation, and EDA for the Kaggle
arXiv subset.

## Standalone use (module owners)

```text
pip install -r requirements.txt   # from the repository root
cd modules/dataset
# Notebooks: 01_data_preprocessing_2.ipynb, 03_eda_1.ipynb
```

## Canonical artifacts

- Base corpus: `data/processed/dev_5k.jsonl`
- Enriched corpus: `data/processed/dev_5k_enriched.jsonl`
- Production retrieval corpus: `data/processed/dev_5k_balanced.jsonl`
- Balanced-corpus audit:
  `results/data_validation/balanced_corpus_report.json`
- Traditional classifier evaluation:
  `results/classification/{metrics.json,model_comparison.csv,*confusion_matrix*}`
- Enrichment evidence:
  `results/data_validation/s2_enrichment_summary.{json,md}`

The completed 9 June 2026 enrichment run processed all 5,000 rows and matched
4,797 Semantic Scholar records. The raw Kaggle snapshot and per-paper API cache
are local, ignored inputs under:

- `data/raw/arxiv-metadata-oai-snapshot.json`
- `data/cache/s2/`

Do not recreate obsolete duplicate copies of this module elsewhere in the repository.
The production retrieval corpus is rebuilt locally from the raw snapshot:

```text
python scripts/build_balanced_corpus.py \
  --raw data/raw/arxiv-metadata-oai-snapshot.json \
  --out data/processed/dev_5k_balanced.jsonl \
  --report results/data_validation/balanced_corpus_report.json
```

It balances `cs.AI`, `cs.CL`, `cs.LG`, and `stat.ML` and uses deterministic
reservoir sampling across foundational, established, recent, and current time
buckets. This mitigates the known file-order bias in `dev_5k.jsonl`.

Evaluate the traditional supervised baseline on a deterministic stratified
held-out split:

```text
python scripts/evaluate_domain_classifier.py \
  --corpus data/processed/dev_5k_balanced.jsonl \
  --output-dir results/classification
```

This compares TF-IDF plus logistic regression against TF-IDF plus Linear SVM on
the same split, producing accuracy, macro/weighted F1, per-class metrics, top
weighted terms, and confusion-matrix artifacts. It is an internal held-out
benchmark; it does not claim independent external validation, multi-label
classification, or a SPECTER2 classification result.

## Full system

Run from the repository root:

```text
python rpa.py run --corpus modules/dataset/data/processed/dev_5k_balanced.jsonl
```
