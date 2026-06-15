# Session 20260613-143122-cc0799eb

- Events: 46
- Components: integration, llm, pdf_nlp
- Failures: 0

## Timeline

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | analyze-pdf | started | Starting analyze-pdf |  |
| integration | parse uploaded PDF | completed | parse uploaded PDF |  |
| pdf_nlp | subprocess | started | Starting pdf_nlp subprocess |  |
| pdf_nlp | analyze_paper | started | Starting production PDF-NLP analysis |  |
| pdf_nlp | parse | completed | parse complete |  |
| pdf_nlp | pos | started | pos started |  |
| pdf_nlp | pos | completed | pos completed | 3572.4 ms |
| pdf_nlp | ner | started | ner started |  |
| pdf_nlp | ner | completed | ner completed | 7726.9 ms |
| pdf_nlp | keyphrases | started | keyphrases started |  |
| pdf_nlp | keyphrases | completed | keyphrases completed | 4393.7 ms |
| pdf_nlp | extractive_summary | started | extractive summary started |  |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 997.7 ms |
| pdf_nlp | structural_checks | started | structural checks started |  |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.4 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| pdf_nlp | analyze_paper | completed | user output |  |
| pdf_nlp | subprocess | completed | pdf_nlp subprocess finished | 19074.7 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| pdf_nlp | parse | completed | PDF parsing completed |  |
| integration | session_config | info | Session configuration |  |
| integration | analyze-pdf START | completed | analyze-pdf START |  |
| integration | skipping related-paper retrieval | completed | skipping related-paper retrieval |  |
| integration | synthesizing summary/findings/gaps | completed | synthesizing summary/findings/gaps |  |
| llm | subprocess | started | Starting llm subprocess |  |
| llm | synthesis | started | user input |  |
| llm | synthesis | completed | synthesis | 105896.9 ms |
| llm | synthesis | completed | user output |  |
| llm | subprocess | completed | llm subprocess finished | 106102.1 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | peer-review assistance | completed | peer-review assistance |  |
| llm | subprocess | started | Starting llm subprocess |  |
| llm | synthesis | started | user input |  |
| llm | synthesis | completed | synthesis | 109065.1 ms |
| llm | synthesis | completed | user output |  |
| llm | subprocess | completed | llm subprocess finished | 109234.3 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | analyze-pdf DONE | completed | analyze-pdf DONE |  |
| integration | output | completed | Run outputs written |  |
| integration | session_complete | completed | Session completed |  |
