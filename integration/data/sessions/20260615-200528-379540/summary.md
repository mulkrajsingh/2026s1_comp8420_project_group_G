# Session 20260615-200528-379540

- State: completed
- Turns: 2
- Messages: 4
- Events: 17
- Components: integration, llm, retrieval
- Failures: 0

## Session Events

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | session_start | started | Session started |  |
| integration | session_complete | completed | Session completed |  |

## Turn 20260615-204935-447326

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | chat | started | Starting chat |  |
| integration | input | completed | User message recorded |  |
| integration | routing | completed | User input analysed |  |
| integration | routing | completed | Selected direct_llm_chat route |  |
| llm | synthesis | completed | synthesis | 6161.6 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant response recorded |  |

## Turn 20260615-204954-731009

| Component | Phase | Status | Message | Duration |
| --- | --- | --- | --- | ---: |
| integration | chat | started | Starting chat |  |
| integration | input | completed | User message recorded |  |
| integration | routing | completed | User input analysed |  |
| integration | routing | completed | Selected retrieval_augmented_chat route |  |
| retrieval | retrieval | completed | retrieval |  |
| llm | synthesis | completed | synthesis | 35143.6 ms |
| integration | output | completed | Run outputs recorded |  |
| integration | output | completed | Assistant RAG response recorded |  |
