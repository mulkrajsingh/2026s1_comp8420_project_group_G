# Session 20260619-200124-119469

- State: completed
- Turns: 1
- Messages: 2
- Events: 19
- Components: integration, llm, pdf_nlp, retrieval
- Failures: 0

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260619-200124-119329

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_with_related_papers route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 3768.5 ms |
| pdf_nlp | ner | completed | ner completed | 8048.2 ms |
| pdf_nlp | keyphrases | completed | keyphrases completed | 4288.2 ms |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 1138.8 ms |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.2 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | retrieve | completed | retrieve |  |
| retrieval | retrieval | completed | retrieval |  |
| integration | synthesize | completed | synthesize |  |
| llm | synthesis | completed | synthesis | 233220.0 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant analysis result recorded |  |
