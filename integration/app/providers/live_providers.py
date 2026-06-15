"""Production providers that invoke member-module CLIs as subprocesses.

Each provider runs a module CLI in its own working directory, validates required
output files, and maps JSON results into the integration contract types.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from ..contracts import (
    PaperRecord,
    ParsedPaper,
    RagEvidencePack,
    Recommendation,
)
from ..session_log import SessionLogger


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _integration_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _retrieval_root() -> Path:
    return _repo_root() / "modules" / "retrieval"


def _llm_root() -> Path:
    return _repo_root() / "modules" / "llm"


def _run_cli(command, cwd, label, required_outputs, session=None):
    """Run one module CLI and enforce its artifact contract."""
    from ..request_control import (
        RequestCancelledError,
        is_cancelled,
        set_process,
    )

    started = time.time()
    if session:
        session.log_subprocess_start(command=command, cwd=str(cwd))
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    set_process(proc)
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if is_cancelled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    if session:
                        session.log_subprocess_end(
                            command=command,
                            returncode=proc.returncode or -1,
                            duration_ms=(time.time() - started) * 1000,
                        )
                    raise RequestCancelledError(f"{label} cancelled")
    finally:
        set_process(None)
    if session:
        session.log_subprocess_end(
            command=command,
            returncode=proc.returncode,
            duration_ms=(time.time() - started) * 1000,
        )
    if proc.returncode != 0:
        detail = stderr or stdout
        raise RuntimeError(
            f"{label} failed (exit {proc.returncode}):\n{detail}"
        )

    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{label} did not create required outputs: {', '.join(missing)}"
        )
    if session:
        for path in required_outputs:
            session.log_artifact_written(str(path))
    return proc


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _log_generation(
    session,
    generation,
    *,
    evidence_ids=None,
):
    """Reject failed generations; synthesis metadata is logged by the LLM CLI."""
    if generation.get("error"):
        raise RuntimeError(
            "LLM generation failed: "
            f"{generation['error']} (backend={generation.get('backend')}, "
            f"model={generation.get('model')})"
        )


def _compose_topic_chat_answer(synthesis: dict) -> str:
    """Build a reader-facing chat answer from a topic-synthesis result.

    The topic-synthesis prompt returns a short ``summary`` plus evidence-backed
    ``key_findings`` and ``research_gaps``. Surfacing only ``summary`` is fragile:
    smaller local models sometimes fill it with a meta restatement of the request
    while the real, source-grounded content lands in ``key_findings``. Combining the
    fields gives the user the substantive answer regardless of which slot the model
    used.
    """
    summary = str(synthesis.get("summary") or "").strip()
    key_findings = [str(item).strip() for item in (synthesis.get("key_findings") or []) if str(item).strip()]
    research_gaps = [str(item).strip() for item in (synthesis.get("research_gaps") or []) if str(item).strip()]

    parts: list[str] = []
    if summary:
        parts.append(summary)
    if key_findings:
        parts.append("**Key findings:**")
        parts.extend(f"- {finding}" for finding in key_findings)
    if research_gaps:
        parts.append("**Research gaps:**")
        parts.extend(f"- {gap}" for gap in research_gaps)

    return "\n\n".join(parts).strip() or summary


def _pdf_nlp_root() -> Path:
    return _repo_root() / "modules" / "pdf_nlp"


def _pdf_nlp_python() -> str:
    """Prefer the pdf_nlp module venv when present."""
    configured = os.environ.get("COMP8420_PDF_NLP_PYTHON")
    if configured:
        # Do not resolve the venv interpreter symlink: resolving it bypasses the
        # virtual environment and loses its installed PDF-NLP dependencies.
        candidate = Path(os.path.abspath(os.path.expanduser(configured)))
        if not candidate.is_file():
            raise FileNotFoundError(
                f"COMP8420_PDF_NLP_PYTHON does not exist: {candidate}"
            )
        return str(candidate)
    for candidate in (
        _pdf_nlp_root() / ".venv" / "bin" / "python",
        _pdf_nlp_root() / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _to_record(d: dict) -> PaperRecord:
    import ast
    authors = d.get("authors", "")
    categories = d.get("categories", "")
    if isinstance(authors, list):
        authors = ", ".join(str(x) for x in authors)
    elif isinstance(authors, str) and authors.strip().startswith("["):
        try:
            authors = ", ".join(str(x) for x in ast.literal_eval(authors))
        except Exception:
            pass
    if isinstance(categories, list):
        categories = " ".join(str(x) for x in categories)
    elif isinstance(categories, str) and categories.strip().startswith("["):
        try:
            categories = " ".join(str(x) for x in ast.literal_eval(categories))
        except Exception:
            pass
    return PaperRecord(
        id=str(d.get("id") or d.get("paper_id") or d.get("arxiv_id") or ""),
        title=(d.get("title") or "").strip(),
        abstract=(d.get("abstract") or "").strip(),
        authors=authors or "",
        categories=categories or "",
        update_date=d.get("update_date") or d.get("published_date"),
    )


def write_corpus_slice(
    corpus_path: Path,
    out_path: Path,
    limit: int | None,
) -> Path:
    """Copy first ``limit`` PaperRecords from dev_5k into POC corpus JSONL."""
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Production corpus missing: {corpus_path} "
            "(expected modules/dataset/data/processed/dev_5k.jsonl)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    effective_limit = None if limit is None or limit <= 0 else limit
    count = 0
    with open(corpus_path) as src, open(out_path, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            count += 1
            if effective_limit is not None and count >= effective_limit:
                break
    if count == 0:
        raise ValueError(f"Production corpus is empty: {corpus_path}")
    return out_path


def raw_recommendations_to_contract(query: str, raw: list) -> Recommendation:
    """Map retrieval-module recommendation JSON into a ``Recommendation``."""
    items = []
    for rec in raw:
        paper = rec.get("paper") or {}
        pid = paper.get("paper_id") or paper.get("id") or ""
        authors = paper.get("authors", [])
        if not isinstance(authors, list):
            authors = [str(authors)] if authors else []
        published = paper.get("published_date") or paper.get("update_date") or ""
        year = published[:4] if isinstance(published, str) and len(published) >= 4 else None
        abstract = (paper.get("abstract") or "").strip()
        snippet = abstract[:300] + ("..." if len(abstract) > 300 else "")
        items.append({
            "paper_id": pid,
            "title": paper.get("title", ""),
            "authors": authors,
            "year": year,
            "url": paper.get("url", ""),
            "score": float(rec.get("score", 0.0)),
            "apa_citation": rec.get("apa_citation", ""),
            "why": rec.get("reason", rec.get("why", "")),
            "snippet": snippet,
        })
    return Recommendation(query=query, items=items)


def evidence_pack_to_rag(pack: dict) -> RagEvidencePack:
    """Convert a retrieval evidence pack dict into ``RagEvidencePack``."""
    snippets = []
    for s in pack.get("evidence_snippets") or []:
        snippets.append({
            "paper_id": s.get("source_id", ""),
            "text": s.get("snippet", ""),
            "score": 1.0,
            "source": pack.get("retrieval_mode", "offline"),
        })
    return RagEvidencePack(
        query=pack.get("query", ""),
        snippets=snippets,
        retrieval_mode=pack.get("retrieval_mode", "offline"),
    )


class SubprocessPdfParser:
    """Run the PDF-NLP parse and enrichment CLI as a subprocess."""

    def __init__(
        self,
        *,
        session: SessionLogger | None = None,
        integration_outputs: Path | None = None,
        module_outputs: Path | None = None,
    ):
        self.session = session
        self.integration_outputs = (
            integration_outputs or _integration_root() / "outputs"
        ).resolve()
        self.module_outputs = (
            module_outputs or _pdf_nlp_root() / "outputs"
        ).resolve()

    def parse(self, pdf_path: str) -> ParsedPaper:
        pdf = Path(pdf_path).resolve()
        if not pdf.is_file():
            raise FileNotFoundError(f"Uploaded PDF not found: {pdf}")
        if pdf.stat().st_size == 0:
            raise ValueError(f"Uploaded PDF is empty: {pdf}")

        cli_path = _pdf_nlp_root() / "app" / "cli.py"
        if not cli_path.is_file():
            raise FileNotFoundError(f"PDF-NLP module missing: {cli_path}")

        self.integration_outputs.mkdir(parents=True, exist_ok=True)
        self.module_outputs.mkdir(parents=True, exist_ok=True)
        out_path = self.integration_outputs / "parsed_paper.json"
        debug_path = self.integration_outputs / "pdf_parse_debug.json"
        analysis_path = self.integration_outputs / "basic_nlp.json"
        review_path = self.integration_outputs / "structural_review.json"
        module_out = self.module_outputs / "parsed_paper.json"
        module_debug = self.module_outputs / "pdf_parse_debug.json"
        module_analysis = self.module_outputs / "basic_nlp.json"
        module_review = self.module_outputs / "structural_review.json"
        for path in (
            out_path,
            debug_path,
            analysis_path,
            review_path,
            module_out,
            module_debug,
            module_analysis,
            module_review,
        ):
            path.unlink(missing_ok=True)

        cmd = [
            _pdf_nlp_python(),
            "-m",
            "app.cli",
            "analyze-paper",
            "--pdf",
            str(pdf),
            "--out",
            str(out_path),
            "--debug-out",
            str(debug_path),
            "--analysis-out",
            str(analysis_path),
            "--review-out",
            str(review_path),
        ]
        _run_cli(
            cmd,
            _pdf_nlp_root(),
            "PDF-NLP analyze-paper",
            (out_path, debug_path, analysis_path, review_path),
            self.session,
        )
        pdf_mtime = pdf.stat().st_mtime
        out_mtime = out_path.stat().st_mtime
        if out_mtime < pdf_mtime - 1:
            raise RuntimeError(
                f"ParsedPaper output is stale for upload: {out_path} "
                f"(output mtime {out_mtime} < pdf mtime {pdf_mtime})"
            )

        parsed = ParsedPaper.from_dict(_read_json(out_path))
        parsed.source_path = str(pdf)
        return parsed


class LivePaperSource:
    """PaperRecord corpus (real JSONL, optional head slice)."""

    def __init__(self, corpus_jsonl: Path):
        self.corpus_jsonl = corpus_jsonl.resolve()
        self._corpus: list[PaperRecord] | None = None

    def get_corpus(self) -> list[PaperRecord]:
        if self._corpus is None:
            if not self.corpus_jsonl.is_file():
                raise FileNotFoundError(f"Production corpus missing: {self.corpus_jsonl}")
            with open(self.corpus_jsonl) as f:
                self._corpus = [_to_record(json.loads(l)) for l in f if l.strip()]
        return list(self._corpus)

    def search_topic(self, query: str, k: int = 5) -> list[PaperRecord]:
        raise RuntimeError(
            "Topic ranking uses the retrieval module recommend-topic command"
        )


class SubprocessRecommender:
    """Run retrieval ``recommend-topic`` on real corpus; load recommendations + RAG pack."""

    def __init__(
        self,
        *,
        query: str,
        corpus_jsonl: Path,
        embedding_model: str | None = None,
        retrieval_strategy: str = "hybrid_rrf",
        top_k: int = 10,
        session: SessionLogger | None = None,
        integration_outputs: Path | None = None,
    ):
        self.query = query
        self.corpus_jsonl = corpus_jsonl.resolve()
        self.embedding_model = embedding_model
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k
        self.session = session
        self.integration_outputs = (
            integration_outputs or _integration_root() / "outputs"
        ).resolve()
        self._ran = False
        self._recommendation: Recommendation | None = None
        self._evidence: RagEvidencePack | None = None

    def _run_retrieval_cli(self) -> None:
        if self._ran:
            return
        cli_path = _retrieval_root() / "app" / "cli.py"
        if not cli_path.is_file():
            raise FileNotFoundError(
                f"Retrieval module missing: {cli_path}"
            )
        if not self.corpus_jsonl.is_file():
            raise FileNotFoundError(
                f"Dataset production corpus missing: {self.corpus_jsonl}"
            )
        self.integration_outputs.mkdir(parents=True, exist_ok=True)
        rec_out = self.integration_outputs / "recommendations.json"
        pack_path = self.integration_outputs / "rag_evidence_pack.json"
        rec_out.unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "app.cli",
            "recommend-topic",
            "--query",
            self.query,
            "--papers",
            str(self.corpus_jsonl),
            "--out",
            str(rec_out),
            "--top-k",
            str(self.top_k),
            "--retrieval-strategy",
            self.retrieval_strategy,
        ]
        if self.embedding_model:
            cmd.extend(["--embedding-model", self.embedding_model])
        _run_cli(
            cmd,
            _retrieval_root(),
            "Retrieval recommend-topic",
            (rec_out, pack_path),
            self.session,
        )
        raw_recs = _read_json(rec_out)
        if not isinstance(raw_recs, list):
            raise ValueError("recommendations.json must be a JSON array")
        self._recommendation = raw_recommendations_to_contract(self.query, raw_recs)
        self._evidence = evidence_pack_to_rag(_read_json(pack_path))
        self._ran = True

    def retrieve_evidence(self, query: str, mode: str = "offline") -> RagEvidencePack:
        if mode != "offline":
            raise ValueError(
                "Retrieval recommend-topic supports only offline local retrieval"
            )
        self._run_retrieval_cli()
        assert self._evidence is not None
        return self._evidence

    def recommend(self, query: str, candidates: list[PaperRecord],
                  k: int = 5) -> Recommendation:
        self._run_retrieval_cli()
        assert self._recommendation is not None
        return self._recommendation

    def search_offline_pack(self, query: str, top_k: int | None = None) -> dict:
        """Run recommend-topic for a ReAct-planned query and return the RAG pack dict."""
        self.query = query
        if top_k is not None:
            self.top_k = top_k
        self._ran = False
        self._recommendation = None
        self._evidence = None
        self._run_retrieval_cli()
        pack_path = self.integration_outputs / "rag_evidence_pack.json"
        return _read_json(pack_path)


class SubprocessSynthesizer:
    """Run LLM ``synthesize`` with Ollama; map JSON output to pipeline fields."""

    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        style: str = "auto",
        recommendation_limit: int = 5,
        query_analysis: dict | None = None,
        prompt_strategy: str = "zero_shot",
        session: SessionLogger | None = None,
        integration_outputs: Path | None = None,
    ):
        self.model = model
        self.style = style
        self.recommendation_limit = recommendation_limit
        self.query_analysis = query_analysis
        self.prompt_strategy = prompt_strategy
        self.session = session
        self.integration_outputs = (
            integration_outputs or _integration_root() / "outputs"
        ).resolve()

    def _run_paper_summary_cli(
        self,
        parsed: ParsedPaper,
        *,
        user_query: str,
        out_name: str = "llm_paper_summary.md",
        command: str = "summarize",
    ) -> dict:
        cli_path = _llm_root() / "app" / "cli.py"
        if not cli_path.is_file():
            raise FileNotFoundError(f"LLM module missing: {cli_path}")
        self.integration_outputs.mkdir(parents=True, exist_ok=True)
        paper_path = self.integration_outputs / "parsed_paper.json"
        md_out = self.integration_outputs / out_name
        json_out = self.integration_outputs / f"{Path(out_name).stem}_result.json"
        metadata_out = self.integration_outputs / f"{Path(out_name).stem}_generation.json"
        for path in (md_out, json_out, metadata_out):
            path.unlink(missing_ok=True)
        with open(paper_path, "w") as f:
            json.dump(parsed.to_dict(), f, indent=2)
        cmd = [
            sys.executable,
            "-m",
            "app.cli",
            command,
            "--paper",
            str(paper_path.resolve()),
            "--query",
            user_query,
            "--style",
            self.style,
            "--out",
            str(md_out.resolve()),
            "--json-out",
            str(json_out.resolve()),
            "--metadata-out",
            str(metadata_out.resolve()),
            "--model",
            self.model,
            "--prompt-strategy",
            self.prompt_strategy,
        ]
        query_analysis_path = self._write_query_analysis()
        if query_analysis_path is not None:
            cmd.extend(["--query-analysis", str(query_analysis_path)])
        _run_cli(
            cmd,
            _llm_root(),
            f"LLM {command}",
            (md_out, json_out, metadata_out),
            self.session,
        )
        payload = _read_json(json_out)
        _log_generation(
            self.session,
            _read_json(metadata_out),
        )
        payload.setdefault("apa_citations", [])
        return payload

    def _run_synthesis_cli(
        self,
        evidence_path: Path,
        *,
        verify: bool = True,
        fast_verify: bool = False,
    ) -> dict:
        cli_path = _llm_root() / "app" / "cli.py"
        if not cli_path.is_file():
            raise FileNotFoundError(f"LLM module missing: {cli_path}")
        if not evidence_path.is_file():
            raise FileNotFoundError(
                f"Retrieval RagEvidencePack missing for LLM synthesis: {evidence_path}"
            )
        self.integration_outputs.mkdir(parents=True, exist_ok=True)
        md_out = self.integration_outputs / "llm_analysis.md"
        json_out = self.integration_outputs / "analysis_result_from_llm.json"
        gen_out = self.integration_outputs / "llm_generation.json"
        for path in (md_out, json_out, gen_out):
            path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "app.cli",
            "synthesize",
            "--evidence",
            str(evidence_path.resolve()),
            "--style",
            self.style,
            "--out",
            str(md_out.resolve()),
            "--json-out",
            str(json_out.resolve()),
            "--model",
            self.model,
            "--prompt-strategy",
            self.prompt_strategy,
        ]
        query_analysis_path = self._write_query_analysis()
        if query_analysis_path is not None:
            cmd.extend(["--query-analysis", str(query_analysis_path)])
        if not verify:
            cmd.append("--no-verify")
        elif fast_verify:
            cmd.append("--fast-verify")
        _run_cli(
            cmd,
            _llm_root(),
            "LLM synthesize",
            (md_out, json_out, gen_out),
            self.session,
        )
        _log_generation(
            self.session,
            _read_json(gen_out),
        )
        llm_result = _read_json(json_out)
        return {
            "summary": llm_result.get("summary", ""),
            "key_findings": llm_result.get("key_findings", []),
            "research_gaps": llm_result.get("research_gaps", []),
            "apa_citations": [],
        }

    def _run_direct_chat_cli(self, question: str) -> str:
        cli_path = _llm_root() / "app" / "cli.py"
        if not cli_path.is_file():
            raise FileNotFoundError(f"LLM module missing: {cli_path}")
        self.integration_outputs.mkdir(parents=True, exist_ok=True)
        response_out = self.integration_outputs / "llm_chat_answer.md"
        metadata_out = self.integration_outputs / "llm_chat_answer_generation.json"
        for path in (response_out, metadata_out):
            path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "app.cli",
            "chat",
            "--query",
            question,
            "--style",
            self.style,
            "--out",
            str(response_out.resolve()),
            "--metadata-out",
            str(metadata_out.resolve()),
            "--model",
            self.model,
            "--prompt-strategy",
            self.prompt_strategy,
        ]
        query_analysis_path = self._write_query_analysis()
        if query_analysis_path is not None:
            cmd.extend(["--query-analysis", str(query_analysis_path)])
        _run_cli(
            cmd,
            _llm_root(),
            "LLM chat",
            (response_out, metadata_out),
            self.session,
        )
        _log_generation(
            self.session,
            _read_json(metadata_out),
            evidence_ids=[],
        )
        return response_out.read_text(encoding="utf-8").strip()

    def _write_query_analysis(self) -> Path | None:
        """Persist upstream routing analysis for child CLI reuse."""
        if self.query_analysis is None:
            return None
        path = self.integration_outputs / "llm_query_analysis.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.query_analysis, handle, indent=2)
        return path.resolve()

    def synthesize(self, parsed, query, evidence, recommendation,
                   style="concise", model_mode="base") -> dict:
        if parsed is not None:
            return self._run_paper_summary_cli(
                parsed,
                user_query=query or "Summarize the uploaded paper.",
            )
        pack_path = self.integration_outputs / "rag_evidence_pack.json"
        return self._run_synthesis_cli(pack_path)

    def peer_review(self, parsed: ParsedPaper, model_mode: str = "base") -> str:
        result = self._run_paper_summary_cli(
            parsed,
            user_query=(
                "Write peer-review style feedback with strengths, weaknesses, "
                "missing evidence, and suggested improvements."
            ),
            out_name="llm_peer_review.md",
            command="review",
        )
        return result["peer_review"]

    def answer(self, parsed, question, evidence) -> str:
        if parsed is not None:
            return self._run_paper_summary_cli(
                parsed,
                user_query=question,
                out_name="llm_chat_answer.md",
            )["summary"]
        pack_path = self.integration_outputs / "rag_evidence_pack.json"
        return _compose_topic_chat_answer(
            self._run_synthesis_cli(pack_path, fast_verify=True)
        )

    def answer_direct(self, question: str) -> str:
        return self._run_direct_chat_cli(question)

    def recommend_papers(self, query: str, recommendation: Recommendation) -> list:
        pack_path = self.integration_outputs / "rag_evidence_pack.json"
        if not pack_path.is_file():
            raise FileNotFoundError(
                f"Retrieval RagEvidencePack missing for recommendation summaries: {pack_path}"
            )
        self.integration_outputs.mkdir(parents=True, exist_ok=True)
        json_out = self.integration_outputs / "paper_recommendation_cards.json"
        metadata_out = (
            self.integration_outputs / "paper_recommendation_generation.json"
        )
        for path in (json_out, metadata_out):
            path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "app.cli",
            "recommend-summaries",
            "--evidence",
            str(pack_path.resolve()),
            "--style",
            self.style,
            "--out",
            str(json_out.resolve()),
            "--metadata-out",
            str(metadata_out.resolve()),
            "--max-papers",
            str(self.recommendation_limit),
            "--model",
            self.model,
            "--prompt-strategy",
            self.prompt_strategy,
        ]
        query_analysis_path = self._write_query_analysis()
        if query_analysis_path is not None:
            cmd.extend(["--query-analysis", str(query_analysis_path)])
        _run_cli(
            cmd,
            _llm_root(),
            "LLM recommend-summaries",
            (json_out, metadata_out),
            self.session,
        )
        _log_generation(
            self.session,
            _read_json(metadata_out),
        )
        payload = _read_json(json_out)
        papers = payload.get("recommended_papers") or []
        if not isinstance(papers, list):
            raise ValueError("paper_recommendation_cards.json must contain a list")
        return papers
