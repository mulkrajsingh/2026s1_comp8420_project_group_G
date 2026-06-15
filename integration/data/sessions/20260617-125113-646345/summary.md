# Session 20260617-125113-646345

- State: completed
- Turns: 1
- Messages: 1
- Events: 15
- Components: integration, pdf_nlp
- Failures: 1

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260617-125121-385843

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_with_related_papers route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 2658.5 ms |
| pdf_nlp | ner | completed | ner completed | 7086.6 ms |
| pdf_nlp | keyphrases | completed | keyphrases completed | 4531.9 ms |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 848.2 ms |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.5 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | retrieve | completed | retrieve |  |
| integration | analyze-pdf | failed | analyze-pdf failed |  |
