#!/usr/bin/env python3
"""Download and install large assets from Google Drive.

Run once after cloning and after pip install. The script fetches zipped model
bundles, extracts them into module paths, and can report what is already present.

    python setup_assets.py              # required runtime and training assets
    python setup_assets.py --optional   # also download raw arXiv and E2E logs
    python setup_assets.py --force      # re-download even if present
    python setup_assets.py --check      # report what is installed
"""

import argparse
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("ERROR: gdown not found. Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Fill in the gdrive_id for each asset after uploading to Google Drive.
# The ID is the string between /d/ and /view in a shareable link:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
# ---------------------------------------------------------------------------
ASSETS = [
    {
        "name": "PDF-NLP runtime models",
        "gdrive_id": "1RgX7zS6gvdIBlcqIY8OwtJu5Cg41WsnG",
        "zip_name": "pdf_nlp_models.zip",
        "extract_to": REPO_ROOT / "modules/pdf_nlp/models/runtime",
        "marker": REPO_ROOT / "modules/pdf_nlp/models/runtime/scier-distilbert-final",
        "size_hint": "~1.2 GB",
        "optional": False,
    },
    {
        "name": "Retrieval embedding index",
        "gdrive_id": "1QbuJ_S--V7AR-1UTavWfrRiX7xffcctO",
        "zip_name": "retrieval_index.zip",
        "extract_to": REPO_ROOT / "modules/retrieval/data/processed/retrieval_index",
        "marker": REPO_ROOT / "modules/retrieval/data/processed/retrieval_index/embeddings/embeddings.npy",
        "size_hint": "~16 MB",
        "optional": False,
    },
    {
        "name": "Qwen3 LoRA GGUF (Ollama model)",
        "gdrive_id": "1SeqKn2M2LarQakfit0Ejj1GeuEWF-8R2",
        "zip_name": "qwen3_gguf.zip",
        "extract_to": REPO_ROOT / "modules/llm/models/ollama/qwen3-research-lora",
        "marker": REPO_ROOT / "modules/llm/models/ollama/qwen3-research-lora/qwen3-research-lora-adapter.gguf",
        "size_hint": "~29 MB",
        "optional": False,
    },
    {
        "name": "LoRA training data",
        "gdrive_id": "1PyVCIgMClMnNP5O09rI-FoWqJwuQgj_v",
        "zip_name": "lora_training_data.zip",
        "extract_to": REPO_ROOT / "modules/llm/data/processed",
        "marker": REPO_ROOT / "modules/llm/data/processed/final_dataset/research_lora_train.jsonl",
        "size_hint": "~31 MB zipped (~150 MB extracted)",
        "optional": False,
    },
    {
        "name": "Raw arXiv metadata snapshot",
        "gdrive_id": "1mRscM13fbwvfwj3jw5gCeciP7LjBTqUW",
        "zip_name": "arxiv_raw_snapshot.zip",
        "extract_to": REPO_ROOT / "modules/dataset/data/raw",
        "marker": REPO_ROOT / "modules/dataset/data/raw/arxiv-metadata-oai-snapshot.json",
        "size_hint": "~4.9 GB",
        "optional": True,
    },
    {
        "name": "E2E test logs",
        "gdrive_id": "1y2-wvqh3Bk06AFaj_cLiu98auzhos4Qx",
        "zip_name": "e2e_test_logs.zip",
        "extract_to": REPO_ROOT,
        "marker": REPO_ROOT / "tests/logs",
        "size_hint": "~29 MB",
        "optional": True,
    },
]


def is_present(asset: dict) -> bool:
    marker: Path = asset["marker"]
    if marker.is_file():
        return True
    if marker.is_dir() and any(marker.iterdir()):
        return True
    dest: Path = asset["extract_to"]
    return dest.exists() and any(dest.iterdir())


def download_and_extract(asset: dict, force: bool = False) -> bool:
    dest: Path = asset["extract_to"]
    zip_path: Path = REPO_ROOT / asset["zip_name"]

    if asset["gdrive_id"].startswith("PLACEHOLDER"):
        print(f"  [SKIP] {asset['name']}: GDrive ID not set yet")
        return False

    if is_present(asset) and not force:
        print(f"  [OK]   {asset['name']} — already present")
        return True

    print(f"\n  Downloading {asset['name']} ({asset['size_hint']}) ...")
    url = f"https://drive.google.com/uc?id={asset['gdrive_id']}"
    try:
        gdown.download(url, str(zip_path), quiet=False)
    except Exception as exc:
        print(f"  [ERROR] Download failed: {exc}")
        return False

    if asset["zip_name"] == "pdf_nlp_models.zip":
        print(f"  Installing to {dest.relative_to(REPO_ROOT)} ...")
        sys.path.insert(0, str(REPO_ROOT / "modules" / "pdf_nlp"))
        try:
            from model_assets import install_model_archive

            install_model_archive(zip_path)
        except Exception as exc:
            print(f"  [ERROR] PDF-NLP model install failed: {exc}")
            return False
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting to {dest.relative_to(REPO_ROOT)} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    zip_path.unlink(missing_ok=True)
    print(f"  [DONE] {asset['name']}")
    return True


def check_assets(include_optional: bool = True) -> None:
    print("Asset status:\n")
    required_ok = True
    for asset in ASSETS:
        if asset.get("optional") and not include_optional:
            continue
        tag = "optional" if asset.get("optional") else "required"
        if asset["gdrive_id"].startswith("PLACEHOLDER"):
            status = "PLACEHOLDER — id not set"
            if not asset.get("optional"):
                required_ok = False
        elif is_present(asset):
            status = "PRESENT"
        else:
            status = "MISSING"
            if not asset.get("optional"):
                required_ok = False
        print(f"  [{status:<22}] ({tag}) {asset['name']}")
    print()
    if not required_ok:
        print("Run `python setup_assets.py` to download missing required assets.")
    if include_optional:
        print("Run `python setup_assets.py --optional` for raw arXiv snapshot and E2E logs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--optional", action="store_true", help="Also download optional bundles")
    parser.add_argument("--check", action="store_true", help="Only report asset status")
    args = parser.parse_args()

    print("=== Research Paper Assistant — Asset Setup ===\n")

    if args.check:
        check_assets(include_optional=True)
        return

    selected = [a for a in ASSETS if not a.get("optional") or args.optional]
    results = [download_and_extract(a, force=args.force) for a in selected]

    print("\n--- Next steps ---")
    print("1. Install Ollama:       https://ollama.ai/download")
    print("2. Pull base model:      ollama pull qwen3:8b")
    print("3. Build LoRA model:     python modules/llm/scripts/build_ollama_research_lora_model.py")
    print("4. Validate PDF models:  cd modules/pdf_nlp && python -m app.cli model-assets && cd ../..")
    print("5. Run the web app:      scripts/rpa web   →   http://127.0.0.1:8000")
    print("6. Evidence map:         docs/REPRODUCIBILITY.md")

    required_results = [r for a, r in zip(selected, results) if not a.get("optional")]
    if not all(required_results):
        print("\nWARNING: Some required assets could not be downloaded (see [SKIP]/[ERROR] above).")
        print("Fill in the gdrive_id values in setup_assets.py and re-run.")


if __name__ == "__main__":
    main()
