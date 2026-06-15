"""Focused tests for PDF-NLP enrichment, optional sections, and model setup."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from model_assets import ModelAssetError, install_model_archive, validate_model_assets
from paper_analysis import analyze_parsed_paper, structural_checks
from pdf_parser import split_sections


def _paper() -> dict:
    artifact = REPO_ROOT / "tests" / "papers" / "artifacts" / "drq_v2_parsed.json"
    return json.loads(artifact.read_text(encoding="utf-8"))


class PaperAnalysisTests(unittest.TestCase):
    def test_optional_sections_are_preserved(self) -> None:
        sections = split_sections(
            "Abstract\nA useful abstract.\n"
            "1 Introduction\nIntro.\n"
            "2 Related Work\nPrior studies.\n"
            "3 Method\nMethod body.\n"
            "4 Results\nResults body.\n"
            "5 Limitations\nKnown limitation.\n"
            "6 Discussion\nDiscussion body.\n"
            "7 Conclusion\nConclusion body.\n"
        )
        self.assertIn("related_work", sections)
        self.assertIn("limitations", sections)
        self.assertIn("discussion", sections)

    def test_structural_checks_are_deterministic(self) -> None:
        checks = structural_checks(_paper())
        self.assertEqual(checks, structural_checks(_paper()))
        self.assertTrue(all("code" in item for item in checks))

    def test_enrichment_populates_contract_fields(self) -> None:
        pos = {"token_count": 2, "tokens": [], "noun_chunks": [], "pos_counts": {}}
        mentions = [
            {
                "text": "Transformer",
                "type": "Method",
                "score": 1.0,
                "source": "baseline_gazetteer",
                "section": "abstract",
                "start": 0,
                "end": 11,
            }
        ]
        keyphrases = [
            {
                "text": "transformer model",
                "score": 0.8,
                "section": "abstract",
                "source": "nadiyah_keybert_minilm",
            }
        ]
        summary = {
            "text": "Source sentence.",
            "sentences": [],
            "candidate_sentence_count": 1,
            "source_traceable": True,
        }
        with (
            patch("paper_analysis.require_model_assets"),
            patch("paper_analysis.pos_analysis", return_value=pos),
            patch("paper_analysis.entity_analysis", return_value=mentions),
            patch("paper_analysis.keyphrase_analysis", return_value=keyphrases),
            patch("paper_analysis.extractive_summary", return_value=summary),
            patch("paper_analysis.structural_checks", return_value=[]),
        ):
            enriched, analysis = analyze_parsed_paper(_paper())
        self.assertEqual(enriched["keywords"], ["transformer model"])
        self.assertEqual(enriched["entities"]["methods"], ["Transformer"])
        self.assertEqual(enriched["analysis"], analysis)
        self.assertEqual(
            analysis["provenance"]["bart_status"],
            "historical comparison only; not production",
        )


class ModelAssetTests(unittest.TestCase):
    def test_manifest_validation_reports_missing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "name": "missing",
                                "path": "missing-model",
                                "required_files": [{"path": "config.json"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = validate_model_assets(root=root / "runtime", manifest_path=manifest)
            self.assertFalse(result["valid"])
            self.assertTrue(result["errors"])

    def test_archive_installer_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(ModelAssetError, "Unsafe archive member"):
                install_model_archive(archive_path, root=Path(temp_dir) / "runtime")


if __name__ == "__main__":
    unittest.main()
