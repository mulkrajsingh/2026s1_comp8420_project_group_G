"""E2E: LLM summarize, synthesize, peer-review on real parsed papers (Ollama)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.harness.bootstrap_artifacts import main as bootstrap_main  # noqa: E402
from tests.harness.logging import TestRunLogger  # noqa: E402
from tests.harness.ollama import require_ollama_or_skip  # noqa: E402
from tests.harness.paths import ARTIFACTS_DIR, parsed_artifact_path, rag_pack_artifact_path  # noqa: E402
from tests.harness.runners import run_llm_summarize, run_llm_synthesize  # noqa: E402


def _ensure_artifacts() -> None:
  if not parsed_artifact_path("drq_v2").is_file():
    code = bootstrap_main()
    if code != 0:
      raise RuntimeError(f"bootstrap_artifacts failed with exit {code}")


@unittest.skipUnless(
  __import__("tests.harness.ollama", fromlist=["ollama_available"]).ollama_available(),
  "Ollama required",
)
class LlmE2ETests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    require_ollama_or_skip()
    _ensure_artifacts()

  def test_paper_only_summarize(self) -> None:
    parsed = parsed_artifact_path("drq_v2")
    out = ARTIFACTS_DIR / "llm_paper_summary.md"
    with TestRunLogger.create("llm_summarize") as run:
      proc = run_llm_summarize(
        parsed,
        out,
        query="Summarize the core contribution and method.",
        env=run.session_env(),
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
      text = out.read_text(encoding="utf-8")
      self.assertGreater(len(text), 200)
      self.assertNotIn("[mock-", text.lower())
      meta = ARTIFACTS_DIR / "llm_paper_summary_generation.json"
      self.assertTrue(meta.is_file())
      gen = json.loads(meta.read_text(encoding="utf-8"))
      self.assertEqual(gen.get("backend"), "ollama")

  def test_rag_synthesize(self) -> None:
    pack = rag_pack_artifact_path("drq_v2")
    self.assertTrue(pack.is_file(), "run bootstrap_artifacts first")
    out = ARTIFACTS_DIR / "llm_analysis.md"
    with TestRunLogger.create("llm_synthesize") as run:
      proc = run_llm_synthesize(pack, out, env=run.session_env())
      self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
      text = out.read_text(encoding="utf-8")
      self.assertGreater(len(text), 200)
      session_text = run.session_path.read_text(encoding="utf-8")
      self.assertIn("synthesis", session_text)

  def test_peer_review_style_summarize(self) -> None:
    parsed = parsed_artifact_path("siga")
    out = ARTIFACTS_DIR / "llm_peer_review.md"
    with TestRunLogger.create("llm_peer_review") as run:
      proc = run_llm_summarize(
        parsed,
        out,
        query=(
          "Write peer-review style feedback with strengths, weaknesses, "
          "and suggestions for improvement."
        ),
        style="reviewer",
        env=run.session_env(),
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
      self.assertGreater(len(out.read_text(encoding="utf-8")), 200)


if __name__ == "__main__":
  unittest.main()
