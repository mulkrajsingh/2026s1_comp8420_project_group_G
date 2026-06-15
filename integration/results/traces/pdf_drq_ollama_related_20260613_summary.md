# Session 20260613-143524-4f2a939e

- Events: 56
- Components: integration, llm, pdf_nlp, retrieval
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
| pdf_nlp | pos | completed | pos completed | 3618.3 ms |
| pdf_nlp | ner | started | ner started |  |
| pdf_nlp | ner | completed | ner completed | 8288.3 ms |
| pdf_nlp | keyphrases | started | keyphrases started |  |
| pdf_nlp | keyphrases | completed | keyphrases completed | 4366.8 ms |
| pdf_nlp | extractive_summary | started | extractive summary started |  |
| pdf_nlp | extractive_summary | completed | extractive summary completed | 1066.0 ms |
| pdf_nlp | structural_checks | started | structural checks started |  |
| pdf_nlp | structural_checks | completed | structural checks completed | 0.6 ms |
| pdf_nlp | analyze_paper | completed | analyze paper completed |  |
| pdf_nlp | analyze_paper | completed | user output |  |
| pdf_nlp | subprocess | completed | pdf_nlp subprocess finished | 19738.2 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| pdf_nlp | parse | completed | PDF parsing completed |  |
| integration | session_config | info | Session configuration |  |
| integration | artifact | completed | Artifact written |  |
| integration | analyze-pdf START | completed | analyze-pdf START |  |
| integration | retrieving evidence (RAG) | completed | retrieving evidence (RAG) |  |
| retrieval | subprocess | started | Starting retrieval subprocess |  |
| retrieval | retrieval | started | user input |  |
| retrieval | retrieval | completed | retrieval |  |
| retrieval | retrieval | completed | user output |  |
| retrieval | subprocess | completed | retrieval subprocess finished | 1895.0 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| retrieval | retrieval | completed | Related-paper retrieval completed |  |
| integration | recommending related papers | completed | recommending related papers |  |
| integration | synthesizing summary/findings/gaps | completed | synthesizing summary/findings/gaps |  |
| llm | subprocess | started | Starting llm subprocess |  |
| llm | synthesis | started | user input |  |
| llm | synthesis | completed | synthesis | 118073.8 ms |
| llm | synthesis | completed | user output |  |
| llm | subprocess | completed | llm subprocess finished | 118182.2 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | peer-review assistance | completed | peer-review assistance |  |
| llm | subprocess | started | Starting llm subprocess |  |
| llm | synthesis | started | user input |  |
| llm | synthesis | completed | synthesis | 85591.4 ms |
| llm | synthesis | completed | user output |  |
| llm | subprocess | completed | llm subprocess finished | 85961.5 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | analyze-pdf DONE | completed | analyze-pdf DONE |  |
| integration | output | completed | Run outputs written |  |
| integration | session_complete | completed | Session completed |  |
