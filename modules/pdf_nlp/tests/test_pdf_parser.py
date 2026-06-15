"""Parser unit tests for canonical ParsedPaper output."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pdf_parser import (  # noqa: E402
    PdfParserError,
    build_parsed_paper,
    extract_references,
    stable_paper_id,
    validate_pdf_path,
)

LLM_SCHEMAS = MODULE_ROOT.parent / "llm" / "app" / "schemas.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("llm_schemas", LLM_SCHEMAS)
assert _spec and _spec.loader
_schemas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schemas)
validate_parsed_paper = _schemas.validate_parsed_paper

SAMPLE_PDF = MODULE_ROOT.parents[1] / "tests" / "papers" / "drq_v2" / "2107.09645v1.pdf"


class PdfParserTests(unittest.TestCase):
    def test_parse_sample_pdf_matches_contract(self) -> None:
        parsed, debug = build_parsed_paper(SAMPLE_PDF)
        validate_parsed_paper(parsed)
        self.assertEqual(parsed["metadata"]["source"], "uploaded_pdf")
        self.assertEqual(parsed["metadata"]["arxiv_id"], "2107.09645v1")
        for key in ("abstract", "introduction", "method", "results", "conclusion"):
            self.assertIn(key, parsed["sections"])
        self.assertTrue(parsed["sections"]["abstract"])
        self.assertTrue(parsed["sections"]["introduction"])
        self.assertGreater(debug["page_count"], 0)
        self.assertTrue(debug["page_text"])

    def test_missing_sections_are_empty_strings(self) -> None:
        parsed, _ = build_parsed_paper(SAMPLE_PDF)
        for key, value in parsed["sections"].items():
            self.assertIsInstance(value, str)

    def test_stable_paper_id_from_filename(self) -> None:
        first = stable_paper_id(SAMPLE_PDF)
        second = stable_paper_id(SAMPLE_PDF)
        self.assertEqual(first, second)
        self.assertIn("2107", first)

    def test_missing_pdf_raises_clear_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            validate_pdf_path(Path("does-not-exist.pdf"))

    def test_empty_pdf_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty = Path(temp_dir) / "empty.pdf"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(PdfParserError, "empty"):
                validate_pdf_path(empty)

    def test_invalid_pdf_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "invalid.pdf"
            invalid.write_text("not a pdf", encoding="utf-8")
            with self.assertRaisesRegex(PdfParserError, "Invalid|Cannot open"):
                validate_pdf_path(invalid)

    def test_cli_round_trip_json(self) -> None:
        parsed, _ = build_parsed_paper(SAMPLE_PDF)
        encoded = json.dumps(parsed)
        round_trip = json.loads(encoded)
        validate_parsed_paper(round_trip)

    def test_bracket_numbered_reference_extraction(self) -> None:
        text = """
        References
        [1] A. Author. Title one. Journal, 2020.
        [2] B. Writer. Title two. Conference, 2021.
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 2)
        self.assertIn("Author", refs[0])
        self.assertIn("Writer", refs[1])

    def test_dot_numbered_reference_extraction(self) -> None:
        text = """
        Bibliography
        1. A. Author. Title one. Journal, 2020.
        2. B. Writer. Title two. Conference, 2021.
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 2)
        self.assertFalse(refs[0].startswith("1."))
        self.assertFalse(refs[1].startswith("2."))

    def test_one_line_author_year_reference_extraction(self) -> None:
        text = """
        References
        Smith, J., and Doe, A. Learning from pixels. NeurIPS, 2019.
        Brown, T., and Green, R. Reinforcement learning at scale. ICML, 2020.
        """
        refs = extract_references(text)
        self.assertGreaterEqual(len(refs), 2)
        self.assertTrue(all(len(ref) < 300 for ref in refs))
        self.assertIn("Smith", refs[0])
        self.assertIn("Brown", refs[1])

    def test_multiline_author_year_reference_extraction(self) -> None:
        text = """
        References
        Smith, J., and Doe, A. Learning from pixels.
        Advances in Neural Information Processing Systems, 2019.
        Brown, T., and Green, R. Reinforcement learning
        at scale. ICML, 2020.
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 2)
        self.assertIn("Advances in Neural Information Processing Systems", refs[0])
        self.assertIn("at scale", refs[1])

    def test_year_suffixes_create_separate_references(self) -> None:
        text = """
        References
        Timothy P. Lillicrap and David Silver. Continuous control. CoRR, 2015a.
        Timothy P. Lillicrap and Daan Wierstra. Deep control. CoRR, 2015b.
        Volodymyr Mnih et al. Playing Atari. arXiv, 2013.
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 3)
        self.assertIn("2015a", refs[0])
        self.assertIn("2015b", refs[1])
        self.assertIn("Mnih", refs[2])

    def test_names_accents_initials_hyphens_apostrophes_and_et_al(self) -> None:
        text = """
        References
        Rémi Munos, Anna Harutyunyan, and Marc G. Bellemare. Safe learning. CoRR, 2016.
        O'Connor, J.-P., et al. Robust evaluation. Journal of AI, (2020).
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 2)
        self.assertIn("Rémi Munos", refs[0])
        self.assertIn("O'Connor", refs[1])

    def test_reference_metadata_is_preserved(self) -> None:
        text = """
        References
        Smith, J., and Doe, A. Useful paper. In ICML, 2020.
        DOI: 10.1000/example https://example.org/paper
        Brown, T., et al. Preprint. arXiv:2101.00001, 2021.
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 2)
        self.assertIn("10.1000/example", refs[0])
        self.assertIn("https://example.org/paper", refs[0])
        self.assertIn("arXiv:2101.00001", refs[1])

    def test_standalone_page_numbers_are_removed(self) -> None:
        text = """
        References
        Smith, J. First paper. Journal, 2019.
        14
        Brown, T. Second paper. Conference, 2020.
        """
        refs = extract_references(text)
        self.assertEqual(len(refs), 2)
        self.assertNotIn(" 14 ", f" {refs[0]} ")
        self.assertNotIn(" 14 ", f" {refs[1]} ")

    def test_duplicate_references_are_removed(self) -> None:
        text = """
        References
        Smith, J. Repeated paper. Journal, 2019.
        Smith, J. Repeated paper. Journal, 2019.
        """
        refs = extract_references(text)
        self.assertEqual(refs, ["Smith, J. Repeated paper. Journal, 2019"])

    def test_missing_references_section_returns_empty_list(self) -> None:
        self.assertEqual(extract_references("Introduction\nNo bibliography here."), [])

    def test_sample_pdf_references_are_individual_entries(self) -> None:
        parsed, _ = build_parsed_paper(SAMPLE_PDF)
        refs = parsed["references"]
        self.assertGreater(len(refs), 5)
        avg_len = sum(len(ref) for ref in refs) / len(refs)
        self.assertLess(avg_len, 400)
        self.assertLess(max(map(len, refs)), 600)
        self.assertTrue(any("Yarats" in ref for ref in refs))
        lillicrap = [ref for ref in refs if ref.startswith("Timothy P. Lillicrap")]
        mnih = [ref for ref in refs if "Mnih" in ref and ref.startswith(("V olodymyr", "Volodymyr"))]
        munos = [ref for ref in refs if ref.startswith("Rémi Munos")]
        self.assertEqual(len(lillicrap), 2)
        self.assertGreaterEqual(len(mnih), 3)
        self.assertEqual(len(munos), 1)
        self.assertTrue(all("Munos" not in ref for ref in lillicrap + mnih))


if __name__ == "__main__":
    unittest.main()
