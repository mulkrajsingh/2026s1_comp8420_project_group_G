# Session 20260617-155846-750844

- State: completed
- Turns: 1
- Messages: 1
- Events: 8
- Components: integration, pdf_nlp
- Failures: 1

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260617-160401-829765

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_with_related_papers route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 3047.3 ms |
| integration | analyze-pdf | failed | analyze-pdf failed |  |
