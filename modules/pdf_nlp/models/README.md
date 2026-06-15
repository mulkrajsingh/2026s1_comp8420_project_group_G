# PDF-NLP Runtime Models

`runtime/` is local and ignored. Install a user-supplied team archive with:

```bash
python -m app.cli model-assets --archive /absolute/path/to/archive.zip
```

Run the command without `--archive` to validate the current installation.
[`manifest.json`](manifest.json) defines the accepted layout, compatibility
versions, attribution, file sizes, and SHA-256 hashes.

The Google Drive URL is intentionally not hard-coded until the project owner
provides the final archive location.
