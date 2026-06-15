# Session 20260615-222652-753877

- State: completed
- Turns: 1
- Messages: 2
- Events: 17
- Components: integration, llm, pdf_nlp
- Failures: 0

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260615-222721-932778

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_paper_only route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 24228.7 ms |
| pdf_nlp | ner | completed | ner completed | 5412.4 ms |
| pdf_nlp | keyphrases | completed | keyphrases completed | 8114.1 ms |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 614.6 ms |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.6 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | synthesize | completed | synthesize |  |
| llm | synthesis | completed | synthesis | 128992.1 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant analysis result recorded |  |
