# Session 20260617-130710-687367

- State: active
- Turns: 1
- Messages: 2
- Events: 18
- Components: integration, llm, pdf_nlp, retrieval
- Failures: 0

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |

## Turn 20260617-155454-922350

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_with_related_papers route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 2924.0 ms |
| pdf_nlp | ner | completed | ner completed | 7244.7 ms |
| pdf_nlp | keyphrases | completed | keyphrases completed | 4813.8 ms |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 745.0 ms |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.1 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | retrieve | completed | retrieve |  |
| retrieval | retrieval | completed | retrieval |  |
| integration | synthesize | completed | synthesize |  |
| llm | synthesis | completed | synthesis | 147223.3 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant analysis result recorded |  |
