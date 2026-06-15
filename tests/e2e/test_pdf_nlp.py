"""E2E: parse real PDF papers via pdf_nlp CLI."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.harness.logging import TestRunLogger  # noqa: E402
from tests.harness.paths import ARTIFACTS_DIR, PAPER_PDFS  # noqa: E402
from tests.harness.runners import run_parse_pdf  # noqa: E402
from tests.harness.runners import run_analyze_paper  # noqa: E402


class PdfNlpE2ETests(unittest.TestCase):
  def test_parse_real_papers(self) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with TestRunLogger.create("pdf_nlp_parse") as run:
      for name, pdf in PAPER_PDFS.items():
        out = ARTIFACTS_DIR / f"{name}_parsed.json"
        proc = run_parse_pdf(pdf, out, env=run.session_env())
        self.assertEqual(
          proc.returncode,
          0,
          msg=proc.stderr or proc.stdout,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        title = (data.get("metadata") or {}).get("title") or ""
        self.assertTrue(title.strip(), f"{name} missing title")
        if name == "drq_v2":
          refs = data.get("references") or []
          self.assertGreaterEqual(len(refs), 1, "DrQ-v2 should have references")

        session_text = run.session_path.read_text(encoding="utf-8")
        self.assertIn("parse_complete", session_text)

  @unittest.skipUnless(
    (
      REPO_ROOT / "modules" / "pdf_nlp" / "models" / "runtime"
      / "scier-distilbert-final" / "model.safetensors"
    ).is_file(),
    "PDF-NLP runtime model archive required",
  )
  def test_full_nlp_analysis_on_real_pdf(self) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with TestRunLogger.create("pdf_nlp_analyze") as run:
      out = ARTIFACTS_DIR / "drq_v2_enriched.json"
      proc = run_analyze_paper(PAPER_PDFS["drq_v2"], out, env=run.session_env())
      self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
      data = json.loads(out.read_text(encoding="utf-8"))
      self.assertTrue(data.get("keywords"))
      self.assertTrue((data.get("analysis") or {}).get("pos"))
      self.assertTrue(
        (data.get("analysis") or {}).get("extractive_summary", {}).get("source_traceable")
      )
      session_text = run.session_path.read_text(encoding="utf-8")
      for phase in ("pos", "ner", "keyphrases", "extractive_summary", "structural_checks"):
        self.assertIn(f'"phase": "{phase}"', session_text)


if __name__ == "__main__":
  unittest.main()
