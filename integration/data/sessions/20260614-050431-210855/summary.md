# Session 20260614-050431-210855

- State: completed
- Turns: 1
- Messages: 2
- Events: 17
- Components: integration, llm
- Failures: 0

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260614-050431-210750

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | chat-pdf | started | Starting chat-pdf |  |
| integration | input | completed | User message recorded |  |
| integration | routing | completed | Selected pdf_grounded_chat route |  |
| integration | session_config | info | Session configuration |  |
| integration | PDF-grounded chat | completed | PDF-grounded chat |  |
| llm | subprocess | started | Starting llm subprocess |  |
| llm | synthesis | started | user input |  |
| llm | synthesis | completed | synthesis | 42531.4 ms |
| llm | synthesis | completed | user output |  |
| llm | subprocess | completed | llm subprocess finished | 55212.3 ms |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | artifact | completed | Artifact written |  |
| integration | output | completed | Run outputs written |  |
| integration | output | completed | Assistant response recorded |  |
