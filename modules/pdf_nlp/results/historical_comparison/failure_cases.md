# Failure Cases

## Original As-Supplied Blockers

- `preprocess.py` has a syntax error.
- PDF parser, CLI, and parser tests fail at import when PyMuPDF4LLM is absent.
- Keyphrase/summary import fails when KeyBERT is absent.
- Real-abstract NER inference fails because the saved tokenizer emits `token_type_ids` for DistilBERT.

## Evaluation Runtime Failures

- None after minimal repair.

## Known Quality Failures

- Nadiyah's parser detects Markdown level-2 headings only; title-like headings can absorb the full paper.
- The SciER model produces fragmented WordPiece entities in some scientific terms.
- BART is abstractive and may paraphrase beyond exact source wording.
- The reference parser uses deterministic heading aliases and approximate bibliography-year counting.
