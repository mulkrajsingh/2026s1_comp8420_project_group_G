# Prompt Library Documentation

## Input Contracts

Paper-only summaries and RAG synthesis are intentionally separate:

- `uploaded_paper_summary` accepts a `ParsedPaper` and `rag_evidence_pack: null`.
  It must not retrieve, cite external evidence, or infer related work.
- `topic_search_synthesis` accepts a `RagEvidencePack` produced upstream. It cites
  source IDs and never fabricates recommendation metadata.

All prompts prohibit invented authors, venues, DOIs, citations, scores, methods,
results, or findings. Renderers add an AI disclosure and backend metadata.

## Query Adaptation

`QueryAnalyzer` produces intent, emotion, topic-specific expertise, verbosity,
style, confidence, style source, and local cosine scores. `--style auto` is the
default. MiniLM embeddings are compared with labeled examples using cosine
similarity. Fields below `0.70` are reranked in one batched pass by a local
TinyBERT semantic cross-encoder. Any explicit CLI style wins for the style field.
The runtime converts the structured analysis into prompt instructions:

- confused beginner requests receive supportive explanations
- frustrated debugging requests receive concise actionable responses
- mathematical or advanced requests receive technical detail
- concise requests remain short
- conversational text is routed to direct local-LLM generation without retrieval

## Prompt Families

| Prompt | Purpose | Stage |
| --- | --- | --- |
| direct text chat | answer conversational text without claiming retrieved evidence | 11 |
| uploaded paper summary | summarize only a supplied parsed PDF | 01, 02 |
| topic search synthesis | synthesize retrieved papers for a topic query | 01, 02 |
| research gap identification | list evidence-supported and assumed gaps separately | 01, 03 |
| citation recommendation | recommend papers with APA strings from metadata only | 01, 03, 05 |
| peer-review critique | produce strengths, weaknesses, missing evidence, improvements | 01, 03 |
| beginner explanation | explain a paper plainly while keeping source IDs | 01 |
| follow-up Q&A | answer from current evidence and state uncertainty | 02 |

## Few-Shot Use

Few-shot examples are limited to tasks where formatting drift is common:

- peer-review critique
- APA citation recommendation
- research gap identification

The comparison command writes `results/prompt_comparison/few_shot_vs_zero_shot.csv`
so the report can show whether few-shot prompting improved structure compliance and
reduced format errors.

## ReAct Tool Calls

The tool-call prompt asks the model to output JSON-like calls such as:

- `search_offline`
- `search_cached_live`
- `fetch_paper_details`

The backend executes these calls. The model never directly accesses the network or
mutates files.

## Self-Verification

The verification pass checks that recommendation claims cite source IDs and that APA
strings come from metadata. Unsupported claims are removed or described as assumptions.
Paper-only summaries instead disclose that no external evidence or retrieval was used.
