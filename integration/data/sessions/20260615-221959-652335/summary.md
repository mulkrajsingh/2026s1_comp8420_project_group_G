# Session 20260615-221959-652335

- State: completed
- Turns: 1
- Messages: 1
- Events: 8
- Components: integration, pdf_nlp
- Failures: 2

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260615-222038-131805

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_paper_only route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | analyze_paper | failed | analyze paper failed |  |
| integration | analyze-pdf | failed | analyze-pdf failed |  |
