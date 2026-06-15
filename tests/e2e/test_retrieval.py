"""E2E: recommend-topic with real corpus and APA citations."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.harness.logging import TestRunLogger  # noqa: E402
from tests.harness.paths import ARTIFACTS_DIR, corpus_path  # noqa: E402
from tests.harness.runners import run_recommend_topic  # noqa: E402


class RetrievalE2ETests(unittest.TestCase):
  def test_recommend_topic_with_apa_citations(self) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "retrieval_recommendations.json"
    query = "reinforcement learning from pixels without reconstruction"

    with TestRunLogger.create("retrieval_recommend") as run:
      proc = run_recommend_topic(
        query,
        corpus_path(sample=True),
        out,
        top_k=10,
        env=run.session_env(),
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

      items = json.loads(out.read_text(encoding="utf-8"))
      self.assertIsInstance(items, list)
      self.assertGreaterEqual(len(items), 3, f"expected >=3 recommendations, got {len(items)}")
      for rec in items[:5]:
        apa = rec.get("apa_citation")
        self.assertTrue(apa and str(apa).strip(), f"missing apa_citation in {rec}")

      pack_path = out.parent / "rag_evidence_pack.json"
      self.assertTrue(pack_path.is_file(), "rag_evidence_pack.json must be written")

      session_text = run.session_path.read_text(encoding="utf-8")
      self.assertIn("retrieval", session_text)


if __name__ == "__main__":
  unittest.main()
