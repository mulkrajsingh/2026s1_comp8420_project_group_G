# Session 20260617-163159-040724

- State: active
- Turns: 1
- Messages: 2
- Events: 16
- Components: integration, llm, pdf_nlp
- Failures: 0

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |

## Turn 20260617-163244-960546

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_paper_only route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 3363.5 ms |
| pdf_nlp | ner | completed | ner completed | 7089.7 ms |
| pdf_nlp | keyphrases | completed | keyphrases completed | 4725.5 ms |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 923.4 ms |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.2 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | synthesize | completed | synthesize |  |
| llm | synthesis | completed | synthesis | 78413.5 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant analysis result recorded |  |
