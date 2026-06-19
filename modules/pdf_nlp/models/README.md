# PDF-NLP Runtime Models

`runtime/` is local and ignored. Install a user-supplied team archive with:

```text
python -m modules.pdf_nlp.app.cli model-assets --archive <path-to-archive.zip>
```

Run the command without `--archive` to validate the current installation.
[`manifest.json`](manifest.json) defines the accepted layout, compatibility
versions, attribution, file sizes, and SHA-256 hashes.

For a fresh clone, the runtime archive is delivered by the repository-root
`setup_assets.py` workflow (`pdf_nlp_models.zip`); see the root `readme.md` for
the one-time download and verification steps. Installing a user-supplied archive
with `--archive` remains supported for module owners.
