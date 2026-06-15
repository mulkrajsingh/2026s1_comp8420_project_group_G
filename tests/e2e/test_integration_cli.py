"""E2E: integration CLI on real papers and prompts (Ollama)."""
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
from tests.harness.paths import (  # noqa: E402
    INTEGRATION_ROOT,
    PROMPTS_DIR,
    parsed_artifact_path,
    paper_path,
)
from tests.harness.runners import run_integration  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _ensure_artifacts() -> None:
    if not parsed_artifact_path("drq_v2").is_file():
        code = bootstrap_main()
        if code != 0:
            raise RuntimeError(f"bootstrap_artifacts failed with exit {code}")


@unittest.skipUnless(
    __import__("tests.harness.ollama", fromlist=["ollama_available"]).ollama_available(),
    "Ollama required",
)
class IntegrationCliE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        require_ollama_or_skip()
        _ensure_artifacts()

    def test_search_topic_from_prompts(self) -> None:
        queries = _load_jsonl(PROMPTS_DIR / "topic_queries.jsonl")
        row = queries[0]
        with TestRunLogger.create("integration_search_topic") as run:
            proc = run_integration(
                "run",
                [
                    "--query",
                    row["query"],
                    "--corpus-limit",
                    "200",
                    "--retrieval-strategy",
                    "tfidf",
                    "--style",
                    row.get("style", "technical"),
                ],
                env=run.session_env(),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            result_path = INTEGRATION_ROOT / "outputs" / "analysis_result.json"
            self.assertTrue(result_path.is_file())
            data = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertTrue((data.get("summary") or "").strip())
            papers = data.get("recommended_papers") or []
            self.assertGreaterEqual(len(papers), 1)
            session_text = run.session_path.read_text(encoding="utf-8")
            self.assertIn("user_input", session_text)
            self.assertIn("outputs_recorded", session_text)

    def test_analyze_pdf_full_pipeline(self) -> None:
        pdf = paper_path("drq_v2")
        with TestRunLogger.create("integration_analyze_pdf") as run:
            proc = run_integration(
                "analyze-pdf",
                [str(pdf)],
                env=run.session_env(),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            trace_path = INTEGRATION_ROOT / "outputs" / "demo_trace.json"
            self.assertTrue(trace_path.is_file())
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIn("steps", trace)

    def test_peer_review_pdf(self) -> None:
        pdf = paper_path("siga")
        with TestRunLogger.create("integration_peer_review") as run:
            proc = run_integration(
                "peer-review",
                [str(pdf)],
                env=run.session_env(),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            peer = INTEGRATION_ROOT / "outputs" / "peer_review.md"
            self.assertTrue(peer.is_file())
            self.assertGreater(len(peer.read_text(encoding="utf-8")), 100)

    def test_chat_with_parsed_paper(self) -> None:
        tasks = _load_jsonl(PROMPTS_DIR / "pdf_tasks.jsonl")
        chat_row = next(r for r in tasks if r.get("task") == "chat")
        parsed = parsed_artifact_path(chat_row["paper"])
        with TestRunLogger.create("integration_chat") as run:
            proc = run_integration(
                "chat",
                [
                    chat_row["query"],
                    "--paper-json",
                    str(parsed),
                ],
                env=run.session_env(),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            self.assertTrue((proc.stdout or "").strip())


if __name__ == "__main__":
    unittest.main()
