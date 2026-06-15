# Session 20260617-123142-670531

- State: active
- Turns: 1
- Messages: 1
- Events: 14
- Components: integration, pdf_nlp
- Failures: 1

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |

## Turn 20260617-124124-375402

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | input | completed | PDF attachment recorded |  |
| integration | routing | completed | Selected pdf_analysis_with_related_papers route |  |
| integration | parse | completed | parse |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | completed | pos completed | 3401.4 ms |
| pdf_nlp | ner | completed | ner completed | 6909.0 ms |
| pdf_nlp | keyphrases | completed | keyphrases completed | 5108.9 ms |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 725.1 ms |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.3 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | retrieve | completed | retrieve |  |
| integration | analyze-pdf | failed | analyze-pdf failed |  |
