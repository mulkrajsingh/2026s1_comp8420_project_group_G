"""Sentence-transformer dense retrieval for scientific papers.

Tries SPECTER2 first, then SPECTER v1, then MiniLM. Saved indexes are matched
to a corpus file via SHA-256 metadata before reuse.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Models tried in order; first one that loads wins
_CANDIDATE_MODELS = [
    "allenai/specter2_base",   # SPECTER2 — SOTA scientific embeddings
    "allenai/specter",          # SPECTER v1
    "all-MiniLM-L6-v2",         # Fast general-purpose fallback
]


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matching_saved_index(
    index_dir: str | Path,
    papers_path: str | Path,
    requested_model: str | None = None,
) -> str | None:
    """Return the saved model name only when the index matches the corpus."""
    index_dir = Path(index_dir)
    try:
        index_meta = json.loads(
            (index_dir / "index_meta.json").read_text(encoding="utf-8")
        )
        model_meta = json.loads(
            (index_dir / "embeddings" / "model_name.json").read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    model_name = str(model_meta.get("model_name") or "")
    if not model_name or (requested_model and requested_model != model_name):
        return None
    if index_meta.get("papers_sha256") != file_sha256(papers_path):
        return None
    if not (index_dir / "embeddings" / "embeddings.npy").is_file():
        return None
    return model_name


def _load_model(model_name: str | None):
    from sentence_transformers import SentenceTransformer

    # An explicit model is an experiment setting: never replace it silently.
    candidates = [model_name] if model_name else _CANDIDATE_MODELS
    for name in candidates:
        try:
            model = SentenceTransformer(name)
            logger.info("Loaded embedding model: %s", name)
            return model, name
        except Exception as exc:
            logger.debug("Could not load %s: %s", name, exc)
    if model_name:
        raise RuntimeError(
            f"Configured embedding model could not be loaded: {model_name}. "
            "Install/cache that exact model before running this ablation."
        )
    raise RuntimeError(
        "No embedding model could be loaded. Install sentence-transformers "
        "and cache one of the configured local models."
    )


class EmbeddingRetriever:
    """Dense semantic retrieval using sentence-transformers (SPECTER2 preferred)."""

    def __init__(self, model_name: str | None = None):
        self._model, self.model_name = _load_model(model_name)
        self._embeddings: np.ndarray | None = None
        self._papers: list[dict] = []

    def fit(self, papers: list[dict], batch_size: int = 32, show_progress: bool = True) -> "EmbeddingRetriever":
        self._papers = list(papers)
        corpus = [f"{p['title']} {p['abstract']}" for p in self._papers]
        self._embeddings = self._model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_emb = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = cosine_similarity(q_emb, self._embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._papers[i], float(scores[i])) for i in top_idx]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self._embeddings)
        with open(directory / "model_name.json", "w") as f:
            json.dump({"model_name": self.model_name, "n_papers": len(self._papers)}, f)

    def load_embeddings(
        self,
        directory: str | Path,
        papers: list[dict] | None = None,
    ) -> None:
        directory = Path(directory)
        self._embeddings = np.load(directory / "embeddings.npy")
        if papers is not None:
            if len(papers) != len(self._embeddings):
                raise ValueError(
                    "Saved embedding count does not match the supplied corpus"
                )
            self._papers = list(papers)

    @property
    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"embedding_{short}"
