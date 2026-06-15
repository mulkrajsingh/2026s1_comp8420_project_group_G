"""Tests for the updated LoRA training-data generators."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lora_dataset.arxiv_rag_labels import (
    build_assistant_label,
    label_citation_recommendation,
    label_topic_synthesis,
)
from lora_dataset.arxiv_rag_retrieval import aps_citation, build_rag_evidence_pack
from lora_dataset.kaggle_corpus import normalize_published_date
from lora_dataset.paper_only import build_paper_only_instructions
from lora_dataset.seeds import build_seed_rows


def _paper() -> dict:
    return {
        "paper_id": "2402.12345",
        "title": "Grounded Retrieval",
        "abstract": "A grounded retrieval method is evaluated on scientific papers.",
        "authors": ["Alice Smith", "Bob Jones", "Carol Lee"],
        "categories": ["cs.CL"],
        "published_date": "2024-02-24",
        "venue": None,
        "doi": None,
        "arxiv_id": "2402.12345",
        "url": "https://arxiv.org/abs/2402.12345",
        "source": "test",
    }


class TrainingDatasetUpdateTests(unittest.TestCase):
    def test_truncated_arxiv_dates_become_iso(self) -> None:
        self.assertEqual(normalize_published_date("Tue, 24 Fe", "2602.21340"), "2026-02-24")
        self.assertEqual(normalize_published_date("Wed, 27 Ma", "1304.1113"), "2013-03-27")
        self.assertEqual(normalize_published_date("Tue, 11 Ju", "2406.07041"), "2024-06-11")

    def test_aps_labels_keep_source_ids(self) -> None:
        distractor_one = {**_paper(), "paper_id": "2", "title": "Vision", "abstract": "Image segmentation."}
        distractor_two = {**_paper(), "paper_id": "3", "title": "Speech", "abstract": "Audio transcription."}
        pack = build_rag_evidence_pack(
            "grounded retrieval",
            [_paper(), distractor_one, distractor_two],
            top_k=1,
        )
        citation = aps_citation(_paper())
        self.assertTrue(citation.startswith("A. Smith et al."))
        self.assertIn("arXiv:2402.12345 (2024)", citation)

        recommendation = label_citation_recommendation(pack)
        self.assertIn("[S1]", recommendation)
        self.assertIn("[1]", recommendation)
        self.assertIn("## References", recommendation)

        synthesis = json.loads(label_topic_synthesis(pack))
        self.assertIn("[S1][1]", synthesis["summary"])

    def test_doi_starting_with_s_is_not_treated_as_source_id(self) -> None:
        paper = {
            **_paper(),
            "doi": "10.1017/S1471068414000106",
            "venue": "Test Journal 12, 1",
        }
        pack = build_rag_evidence_pack(
            "grounded retrieval",
            [
                paper,
                {**_paper(), "paper_id": "2", "title": "Vision", "abstract": "Images."},
                {**_paper(), "paper_id": "3", "title": "Speech", "abstract": "Audio."},
            ],
            top_k=1,
        )
        self.assertIsNotNone(build_assistant_label(pack, "citation_recommendation"))

    def test_paper_only_builder_adds_each_requested_task(self) -> None:
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "Task\n\nPaper text:\n\nSource text."},
                    {"role": "assistant", "content": "Summary."},
                ],
                "source": "scitldr",
                "task": "paper_to_summary",
                "prompt_id": "scitldr_train_only",
                "license_note": "test",
            },
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": (
                            "Title: Test\n\nContext:\n\nEvidence.\n\nQuestion: What is stated?\n\n"
                            "Answer using only the context."
                        ),
                    },
                    {"role": "assistant", "content": "Evidence. [S1]"},
                ],
                "source": "qasper",
                "task": "evidence_to_answer",
                "prompt_id": "qasper_train_only",
                "license_note": "test",
            },
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": "Title: Test\n\nPaper abstract:\n\nAn abstract.",
                    },
                    {"role": "assistant", "content": "Strengths: clear scope."},
                ],
                "source": "peerread",
                "task": "peer_review_critique",
                "prompt_id": "peerread_train_only",
                "license_note": "test",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "output.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            counts = build_paper_only_instructions(
                input_path=source,
                output_path=output,
                per_task=1,
            )
            self.assertEqual(sum(counts.values()), 3)
            with output.open() as handle:
                self.assertEqual(sum(1 for _ in handle), 3)

    def test_train_anchors_do_not_reuse_eval_ids(self) -> None:
        ids = {row["prompt_id"] for row in build_seed_rows()}
        self.assertEqual(len(ids), 6)
        self.assertFalse(ids.intersection({f"P0{i}" for i in range(1, 7)}))


if __name__ == "__main__":
    unittest.main()
