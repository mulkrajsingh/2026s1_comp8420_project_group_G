from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

RETRIEVAL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = RETRIEVAL_ROOT / "app" / "retrieval" / "embeddings.py"
SPEC = importlib.util.spec_from_file_location("retrieval_embeddings_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load embedding module: {MODULE_PATH}")
EMBEDDINGS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EMBEDDINGS)


class EmbeddingSelectionTests(unittest.TestCase):
    def test_explicit_embedding_model_is_not_silently_replaced(self) -> None:
        attempted: list[str] = []
        module = types.ModuleType("sentence_transformers")

        def sentence_transformer(name: str):
            attempted.append(name)
            raise RuntimeError("model unavailable")

        module.SentenceTransformer = sentence_transformer
        with (
            patch.dict(sys.modules, {"sentence_transformers": module}),
            self.assertRaisesRegex(
                RuntimeError,
                "Configured embedding model could not be loaded: custom/model",
            ),
        ):
            EMBEDDINGS._load_model("custom/model")

        self.assertEqual(attempted, ["custom/model"])

    def test_saved_index_requires_matching_corpus_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            papers = root / "papers.jsonl"
            papers.write_text('{"paper_id":"p1"}\n', encoding="utf-8")
            embeddings = root / "index" / "embeddings"
            embeddings.mkdir(parents=True)
            (embeddings / "embeddings.npy").write_bytes(b"index")
            (embeddings / "model_name.json").write_text(
                json.dumps({"model_name": "allenai/specter2_base"}),
                encoding="utf-8",
            )
            (root / "index" / "index_meta.json").write_text(
                json.dumps({"papers_sha256": EMBEDDINGS.file_sha256(papers)}),
                encoding="utf-8",
            )

            self.assertEqual(
                EMBEDDINGS.matching_saved_index(root / "index", papers),
                "allenai/specter2_base",
            )
            self.assertIsNone(
                EMBEDDINGS.matching_saved_index(
                    root / "index",
                    papers,
                    "custom/model",
                )
            )
            papers.write_text('{"paper_id":"changed"}\n', encoding="utf-8")
            self.assertIsNone(
                EMBEDDINGS.matching_saved_index(root / "index", papers)
            )


if __name__ == "__main__":
    unittest.main()
