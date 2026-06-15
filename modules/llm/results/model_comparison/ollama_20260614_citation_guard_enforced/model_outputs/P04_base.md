# P04 citation_recommendation

- Backend: `ollama`
- Model: `qwen3:8b`
- Strategy: `few_shot`
- Latency seconds: `24.1097`
- Error: `none`
- Evidence IDs used: `2102.00002`

## Prompt

```text
You are a local research-paper assistant. Answer only from ParsedPaper fields, RagEvidencePack evidence snippets, Recommendation metadata, or explicitly stated assumptions. Cite source IDs for evidence-backed claims. Do not invent authors, venues, citations, scores, or findings. If evidence is incomplete, say so.

Style: technical.

Task: recommend only citations that are directly relevant to the query, using RagEvidencePack candidates and evidence snippets.
For each recommended paper, copy its supplied APA citation exactly, include its source ID in square brackets, and give one evidence-grounded sentence explaining relevance. Do not recommend weakly related or unrelated candidates merely to fill a list. Briefly identify rejected candidates when that distinction prevents a misleading recommendation. Broad field overlap alone is insufficient: the supplied title or snippet must match the query's core method, problem, or setting. A generic paper sharing only an umbrella term such as reinforcement learning may be labelled a nearby lead, but not a recommended citation. Never infer visual control, a method, or a result from a generic robotics/RL snippet. If none are sufficiently relevant, say so explicitly.
Never fabricate or alter DOI, URL, venue, year, author, title, or findings.

Few-shot examples:

Input: Query: retrieval-grounded question answering. Candidate [A] directly studies retrieval-grounded generation and supplies APA string <APA A>. Candidate [B] studies unrelated image classification.

Output: Recommended: [A] <APA A>. Its supplied evidence directly addresses retrieval-grounded generation. Not recommended: [B], because the supplied evidence is about an unrelated task. Do not replace placeholders or copy example entities into the real answer.

Return the final user-visible answer only. Use compact Markdown sections.

Input JSON:

{
  "task": "citation_recommendation",
  "style": "technical",
  "input": {
    "parsed_paper": null,
    "rag_evidence_pack": {
      "query": "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning",
      "eligibility_rule": "Only eligible_candidates may be recommended. nearby_candidates may be reported as nearby leads, never as direct recommendations.",
      "evidence_snippets": [
        {
          "source_id": "2102.00002",
          "title": "Reinforcement Learning for Robotics",
          "snippet": "An RL method for robotic control is introduced and evaluated.",
          "metadata": {
            "year": "2021",
            "authors": [
              "Alan Turing"
            ],
            "venue": null,
            "doi": "10.1234/xyz",
            "categories": [
              "cs.LG",
              "cs.RO"
            ],
            "citation_count": null,
            "url": "https://arxiv.org/abs/2102.00002"
          }
        },
        {
          "source_id": "2103.00003",
          "title": "Bayesian Methods in Statistical Machine Learning",
          "snippet": "We study Bayesian inference techniques for ML models.",
          "metadata": {
            "year": "2021",
            "authors": [
              "Ada Lovelace"
            ],
            "venue": null,
            "doi": null,
            "categories": [
              "stat.ML",
              "stat.TH"
            ],
            "citation_count": null,
            "url": "https://arxiv.org/abs/2103.00003"
          }
        },
        {
          "source_id": "2101.00001",
          "title": "A Transformer Approach to Neural Machine Translation",
          "snippet": "We propose a new transformer model that outperforms prior work on WMT.",
          "metadata": {
            "year": "2021",
            "authors": [
              "Jane Doe",
              "John Smith"
            ],
            "venue": null,
            "doi": null,
            "categories": [
              "cs.CL",
              "cs.LG"
            ],
            "citation_count": null,
            "url": "https://arxiv.org/abs/2101.00001"
          }
        },
        {
          "source_id": "2104.00004",
          "title": "A General Theory of Artificial Intelligence",
          "snippet": "A unifying framework for AI reasoning systems is proposed.",
          "metadata": {
            "year": "2021",
            "authors": [
              "Geoffrey H"
            ],
            "venue": null,
            "doi": null,
            "categories": [
              "cs.AI"
            ],
            "citation_count": null,
            "url": "https://arxiv.org/abs/2104.00004"
          }
        }
      ],
      "eligible_candidates": [],
      "nearby_candidates": [
        {
          "source_id": "2102.00002",
          "title": "Reinforcement Learning for Robotics",
          "abstract": "An RL method for robotic control is introduced and evaluated.",
          "authors": [
            "Alan Turing"
          ],
          "published_date": "2021-02-02",
          "venue": null,
          "doi": "10.1234/xyz",
          "url": "https://arxiv.org/abs/2102.00002",
          "apa_citation": "Turing, A. (2021). Reinforcement Learning for Robotics. *arXiv*. https://doi.org/10.1234/xyz",
          "query_term_coverage": 0.429,
          "matched_query_terms": [
            "control",
            "learning",
            "reinforcement"
          ],
          "missing_query_terms": [
            "augmented",
            "continuous",
            "data",
            "visual"
          ]
        }
      ],
      "rejected_candidates": [
        {
          "source_id": "2103.00003",
          "title": "Bayesian Methods in Statistical Machine Learning",
          "abstract": "We study Bayesian inference techniques for ML models.",
          "authors": [
            "Ada Lovelace"
          ],
          "published_date": "2021-03-03",
          "venue": null,
          "doi": null,
          "url": "https://arxiv.org/abs/2103.00003",
          "apa_citation": "Lovelace, A. (2021). Bayesian Methods in Statistical Machine Learning. *arXiv*. https://arxiv.org/abs/2103.00003",
          "query_term_coverage": 0.143,
          "matched_query_terms": [
            "learning"
          ],
          "missing_query_terms": [
            "augmented",
            "continuous",
            "control",
            "data",
            "reinforcement",
            "visual"
          ]
        },
        {
          "source_id": "2101.00001",
          "title": "A Transformer Approach to Neural Machine Translation",
          "abstract": "We propose a new transformer model that outperforms prior work on WMT.",
          "authors": [
            "Jane Doe",
            "John Smith"
          ],
          "published_date": "2021-01-01",
          "venue": null,
          "doi": null,
          "url": "https://arxiv.org/abs/2101.00001",
          "apa_citation": "Doe, J., & Smith, J. (2021). A Transformer Approach to Neural Machine Translation. *arXiv*. https://arxiv.org/abs/2101.00001",
          "query_term_coverage": 0.0,
          "matched_query_terms": [],
          "missing_query_terms": [
            "augmented",
            "continuous",
            "control",
            "data",
            "learning",
            "reinforcement",
            "visual"
          ]
        },
        {
          "source_id": "2104.00004",
          "title": "A General Theory of Artificial Intelligence",
          "abstract": "A unifying framework for AI reasoning systems is proposed.",
          "authors": [
            "Geoffrey H"
          ],
          "published_date": "2021-04-04",
          "venue": null,
          "doi": null,
          "url": "https://arxiv.org/abs/2104.00004",
          "apa_citation": "H, G. (2021). A General Theory of Artificial Intelligence. *arXiv*. https://arxiv.org/abs/2104.00004",
          "query_term_coverage": 0.0,
          "matched_query_terms": [],
          "missing_query_terms": [
            "augmented",
            "continuous",
            "control",
            "data",
            "learning",
            "reinforcement",
            "visual"
          ]
        }
      ]
    }
  },
  "expected_output_contract": {
    "must_include_source_ids": true,
    "must_separate_findings_from_assumptions": true,
    "must_not_fabricate_metadata": true
  }
}
```

## Output

No directly relevant citation was found in the supplied evidence.

Nearby leads (not recommended as direct citations):
- [2102.00002] Turing, A. (2021). Reinforcement Learning for Robotics. *arXiv*. https://doi.org/10.1234/xyz. Supported query terms: control, learning, reinforcement. Missing core query terms: augmented, continuous, data, visual.
