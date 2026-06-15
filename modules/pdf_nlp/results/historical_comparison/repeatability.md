# Offline Repeatability

- Semantic outputs matching: 10/10.
- All functional semantic outputs match: `True`.
- Method: SHA-256 over each raw pipeline output after excluding runtime, process-memory, timing, full extracted-text, and the non-functional PyMuPDF4LLM markdown character-count diagnostic.

## Non-Functional Diagnostic Variation

- `bert` / `nadiyah_repaired` `parser.markdown_characters`: 66226 vs 66230. None observed; retained parser sections and all scored NLP outputs match.
- `drq_v2` / `nadiyah_repaired` `parser.markdown_characters`: 55004 vs 55006. None observed; retained parser sections and all scored NLP outputs match.
