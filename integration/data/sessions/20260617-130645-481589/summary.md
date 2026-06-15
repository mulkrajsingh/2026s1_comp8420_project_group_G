# Session 20260617-130645-481589

- State: completed
- Turns: 2
- Messages: 3
- Events: 14
- Components: integration, llm, pdf_nlp
- Failures: 1

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260617-130654-613125

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_with_related_papers route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| integration | analyze-pdf | failed | analyze-pdf failed |  |

## Turn 20260617-130700-223260

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | chat | started | Starting chat |  |
| integration | input | completed | User message recorded |  |
| integration | routing | completed | User input analysed |  |
| integration | routing | completed | Selected direct_llm_chat route |  |
| llm | synthesis | completed | synthesis | 2186.0 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant response recorded |  |
