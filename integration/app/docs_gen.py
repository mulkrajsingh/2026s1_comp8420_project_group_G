"""Generators for submission helper documents under ``outputs/``.

These modules write markdown checklists and demo scripts that stay in the
repository alongside generated analysis artifacts.
"""
from __future__ import annotations

from .io_paths import write_text


def write_video_script() -> str:
    """Write ``outputs/video_demo_script.md`` with a five-minute demo plan."""
    md = """# Video demo script (max 5 minutes)

Cached/offline run so nothing fails on network or model latency. All five members speak.

| Time | Segment | Shown | Speaker |
| --- | --- | --- | --- |
| 0:00-0:30 | Problem + architecture | Slide 2 architecture diagram | Sidharth |
| 0:30-1:10 | Dataset + EDA | Yash's charts, classifier table | Yash |
| 1:10-1:50 | PDF parsing + NER/POS | `parse-pdf` output, NER table | Nadiyah |
| 1:50-2:40 | Retrieval comparison | TF-IDF/BM25/SPECTER2 table, PCA | Retrieval member |
| 2:40-3:30 | Prompting + LoRA | prompt comparison, base vs LoRA | LLM member |
| 3:30-4:30 | Live system demo | `python -m app.cli run` -> analysis_result.json + report | Sidharth |
| 4:30-5:00 | Findings + limitations | key results, future work | Anyone |

## Demo commands to run on screen
```text
python rpa.py run --corpus-limit 200
python rpa.py search-topic "retrieval augmented generation for science"
python rpa.py analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf
```
Show `outputs/demo_report.md` and one evaluation table/chart.

## Safety net
If anything fails live, show `05_end_to_end_demo.ipynb` already executed, plus the
cached `outputs/analysis_result.json`.
"""
    return write_text("video_demo_script.md", md)


def write_packaging_checklist() -> str:
    """Write ``outputs/submission_packaging_checklist.md`` for final packaging."""
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
    04_model_comparison.ipynb                    (LLM module)
    05_end_to_end_demo.ipynb                     (Sidharth)
    app/  tests/  outputs/  results/  README.md
  Individual_contribution_form/
    GroupID_StudentID_Contribution.pdf           (one per member)
  Video/          demo_video.mp4
```
Upload as ONE zip to iLearn.

## Checklist
- [ ] All five notebooks run top-to-bottom without errors
- [ ] README explains setup + how to run `python tests/run_system_tests.py`
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
