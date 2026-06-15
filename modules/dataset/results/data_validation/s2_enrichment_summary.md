# Semantic Scholar Enrichment Summary

The completed 9 June 2026 run processed all 5,000 `PaperRecord` rows:

| Measure | Result |
| --- | ---: |
| Semantic Scholar matches | 4,797 |
| Not found | 203 |
| Match rate | 95.94% |
| Recorded runtime | 6:46:44 |
| Unique cache files | 4,897 |

The canonical corpus was rebuilt from the retained cache after an interrupted
rerun had truncated the previous output. The rebuilt
`data/processed/dev_5k_enriched.jsonl` contains 5,000 rows, of which 4,797 have
`s2_enriched=true`. The remaining rows comprise 100 cached misses and 103 rows
without a unique cache entry.

## Integrity

- Input SHA-256: `e67c59ca939828f3eea063c23d08f28a7c150d2626e255172a54d623135d30c1`
- Enriched SHA-256: `a8757c4df3efcd16b502c61d09d65249e0e8fe41c3a0701ff5a5abc9d80cae34`
- Raw snapshot SHA-256: `7a819aa11b09a817cc8e4930eff269438963930573f6ac9aa7ff9d92c5b1183c`

The raw snapshot and per-paper cache are local reproducibility inputs and are
excluded from Git and the final submission. The processed corpus and this
summary are the stable project evidence.
