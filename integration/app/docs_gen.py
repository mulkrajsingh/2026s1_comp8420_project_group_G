"""Stages 08 & 09 — video demo script + submission packaging checklist.

Both are documents (no heavy code). Generated so they live in the repo and stay
consistent with the rest of the workstream.
"""
from __future__ import annotations

from .io_paths import write_text


def write_video_script() -> str:
    """Stage 08 artifact: outputs/video_demo_script.md (5-minute plan)."""
    md = """# Video demo script (max 5 minutes)

Cached/offline run so nothing fails on network or model latency. All five members speak.

| Time | Segment | Shown | Speaker |
| --- | --- | --- | --- |
| 0:00-0:30 | Problem + architecture | Slide 2 architecture diagram | Sidharth |
| 0:30-1:10 | Dataset + EDA | Yash's charts, classifier table | Yash |
| 1:10-1:50 | PDF parsing + NER/POS | `parse-pdf` output, NER table | Nadiyah |
| 1:50-2:40 | Retrieval comparison | TF-IDF/BM25/SPECTER2 table, PCA | Bank |
| 2:40-3:30 | Prompting + LoRA | prompt comparison, base vs LoRA | Mulkraj |
| 3:30-4:30 | Live system demo | `python -m app.cli run` -> analysis_result.json + report | Sidharth |
| 4:30-5:00 | Findings + limitations | key results, future work | Anyone |

## Demo commands to run on screen
```bash
python -m app.cli run --corpus-limit 200    # topic search with session log
python -m app.cli search-topic "retrieval augmented generation for science"
python -m app.cli analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf
```
Show `outputs/demo_report.md` and one evaluation table/chart.

## Safety net
If anything fails live, show `05_end_to_end_demo.ipynb` already executed, plus the
cached `outputs/analysis_result.json`.
"""
    return write_text("video_demo_script.md", md)


def write_packaging_checklist() -> str:
    """Stage 09 artifact: outputs/submission_packaging_checklist.md."""
    md = """# Submission packaging checklist

## Required ZIP structure (wrong structure = penalty)
```
GroupID_Assignment3/
  Report/         GroupID_Report.pdf            (max 5,000 words)
  Presentation/   Presentation.pdf              (PowerPoint exported to PDF)
  Codes/
    01_data_preprocessing.ipynb                 (Yash)
    02_pdf_basic_nlp.ipynb                       (Nadiyah)
    03_rag_recommendation_evaluation.ipynb       (Bank)
    04_model_comparison.ipynb                    (Mulkraj)
    05_end_to_end_demo.ipynb                     (Sidharth)
    app/  tests/  outputs/  results/  README.md
  Individual_contribution_form/
    GroupID_StudentID_Contribution.pdf           (one per member)
  Video/          demo_video.mp4
```
Upload as ONE zip to iLearn.

## Checklist
- [ ] All five notebooks run top-to-bottom without errors
- [ ] README explains setup + how to run `tests/run_system_tests.sh`
- [ ] Root `requirements.txt` included
- [ ] Small reproducible sample data (<5MB) included
- [ ] Cached demo outputs present (`outputs/analysis_result.json`, `demo_report.md`)
- [ ] Large models: download links in the report, not in the zip
- [ ] Report has architecture diagram + tables AND charts
- [ ] Each member's individual contribution form filled in
- [ ] Video recorded, accessible link in the report
- [ ] GitHub repo set to PRIVATE before deadline, public after
- [ ] Results reproducible on a clean clone

## Deadlines
- Presentation: Friday 5 June 2026 (PowerPoint submitted before)
- Report + Code + Video: Friday 19 June 2026, 11:59:59 pm
"""
    return write_text("submission_packaging_checklist.md", md)
