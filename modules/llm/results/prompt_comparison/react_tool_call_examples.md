# ReAct Tool-Call Prompting Examples

The LLM proposes structured tool calls; the backend remains responsible for execution.

## User Query: Find recent papers extending retrieval augmented generation for scientific literature.

```json
{
  "tool": "search_offline",
  "arguments": {
    "query": "retrieval augmented generation scientific literature",
    "top_k": 10,
    "preferred_categories": [
      "cs.CL",
      "cs.AI",
      "cs.LG"
    ]
  }
}
```

Backend rule: The backend executes the structured call; the model does not fetch papers directly.

## User Query: Add citation counts for the top recommendations if cached live data exists.

```json
{
  "tool": "search_cached_live",
  "arguments": {
    "query": "top recommendation citation counts",
    "cache_namespace": "openalex_semantic_scholar",
    "network_allowed": false
  }
}
```

Backend rule: Cache is checked first; live network use is a separate explicit mode.

## User Query: Show details for arXiv paper 2504.02377.

```json
{
  "tool": "fetch_paper_details",
  "arguments": {
    "paper_id": "arxiv_2504_02377",
    "source": "arxiv_api"
  }
}
```

Backend rule: The detail fetcher returns a PaperRecord-compatible object.
