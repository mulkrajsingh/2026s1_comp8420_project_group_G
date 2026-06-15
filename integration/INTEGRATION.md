# Integration Wiring Reference

Short reference for how capability modules connect through the integration
layer. For the full architecture diagram and contract table, see
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

## Provider lifecycle

1. `integration/app/service.py` creates a fresh `Providers` container for each request.
2. The parser runs before any related-paper provider is configured.
3. Retrieval providers are added only for routes that require them.
4. The container is passed directly to `integration/app/pipeline.py`; no global
   state is preserved between requests.
5. Session events retain `file-backed` and `live` source labels.

## Production subprocess mapping

| Provider | Module CLI |
| --- | --- |
| `SubprocessPdfParser` | `modules/pdf_nlp/app/cli.py analyze-paper` |
| `LivePaperSource` | dataset corpus slice |
| `SubprocessRecommender` | `modules/retrieval/app/cli.py recommend-topic` |
| `SubprocessSynthesizer` | `modules/llm/app/cli.py summarize` or `synthesize` |

## Async job endpoints

Available for automation and observability (the canonical frontend uses
synchronous requests):

- `POST /api/jobs/analyze-pdf`
- `POST /api/jobs/search-topic`
- `GET /api/jobs/{job_id}?after=<cursor>`

## Current limitations

- PDF-NLP evaluation uses five real papers with provisional annotations.
- Use production commands and session evidence for integration claims.
- Runtime `integration/data/sessions/` and `integration/outputs/` are
  generated and gitignored. Move only redacted evidence into
  `integration/results/`.
- Logging details: [`docs/OBSERVABILITY.md`](../docs/OBSERVABILITY.md).
