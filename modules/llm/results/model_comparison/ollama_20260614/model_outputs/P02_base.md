# P02 topic_search_synthesis

- Backend: `ollama`
- Model: `qwen3:8b`
- Strategy: `few_shot`
- Latency seconds: `32.8167`
- Error: `none`
- Evidence IDs used: `2101.00001, 2102.00002, 2103.00003, 2104.00004`

## Prompt

```text
You are a local research-paper assistant. Answer only from ParsedPaper fields, RagEvidencePack evidence snippets, Recommendation metadata, or explicitly stated assumptions. Cite source IDs for evidence-backed claims. Do not invent authors, venues, citations, scores, or findings. If evidence is incomplete, say so.

Style: technical.

Task: synthesize retrieved papers for a topic query.
Use only RagEvidencePack candidates and evidence_snippets. Cite source IDs in each claim.
Respond with a single JSON object (no markdown fences) with keys:
  "summary": string (2-4 sentences),
  "key_findings": array of strings (evidence-backed, with source IDs),
  "research_gaps": array of strings (gaps or limitations supported by evidence).
Do not invent metadata. If evidence is insufficient, state that in summary or gaps.

Return the final answer as a single JSON object only (keys: "summary", "key_findings", "research_gaps"). No markdown fences or extra prose.

Input JSON:

{
  "task": "topic_search_synthesis",
  "style": "technical",
  "input": {
    "parsed_paper": null,
    "rag_evidence_pack": {
      "query": "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning",
      "retrieval_mode": "offline",
      "prompt_strategy": "chain_of_thought",
      "prompt_templates": {
        "zero_shot": "You are a research assistant. Using ONLY the evidence snippets below, answer the query. Cite each claim using the source_id in brackets.\n\nQuery: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning\n\nEvidence:\n[2102.00002] Reinforcement Learning for Robotics: An RL method for robotic control is introduced and evaluated.\n[2103.00003] Bayesian Methods in Statistical Machine Learning: We study Bayesian inference techniques for ML models.\n[2101.00001] A Transformer Approach to Neural Machine Translation: We propose a new transformer model that outperforms prior work on WMT.\n[2104.00004] A General Theory of Artificial Intelligence: A unifying framework for AI reasoning systems is proposed.\n\nAnswer (cite sources):",
        "few_shot": "You are a research assistant. Below are examples of how to answer using evidence.\n\nExample:\nQuery: What is transfer learning?\nEvidence: [S1] Transfer learning reuses a model trained on one task for another...\nAnswer: Transfer learning reuses pretrained models for new tasks [S1].\n\nNow answer the following using ONLY the evidence below:\nQuery: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning\n\nEvidence:\n[2102.00002] Reinforcement Learning for Robotics: An RL method for robotic control is introduced and evaluated.\n[2103.00003] Bayesian Methods in Statistical Machine Learning: We study Bayesian inference techniques for ML models.\n[2101.00001] A Transformer Approach to Neural Machine Translation: We propose a new transformer model that outperforms prior work on WMT.\n[2104.00004] A General Theory of Artificial Intelligence: A unifying framework for AI reasoning systems is proposed.\n\nAnswer (cite sources):",
        "chain_of_thought": "You are a research assistant. Think step by step before answering.\n\nQuery: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning\n\nEvidence:\n[2102.00002] Reinforcement Learning for Robotics: An RL method for robotic control is introduced and evaluated.\n[2103.00003] Bayesian Methods in Statistical Machine Learning: We study Bayesian inference techniques for ML models.\n[2101.00001] A Transformer Approach to Neural Machine Translation: We propose a new transformer model that outperforms prior work on WMT.\n[2104.00004] A General Theory of Artificial Intelligence: A unifying framework for AI reasoning systems is proposed.\n\nStep 1 - Identify the most relevant papers from the evidence.\nStep 2 - Extract key findings from those papers.\nStep 3 - Synthesise a coherent answer citing each source.\n\nAnswer:"
      },
      "candidates": [
        {
          "paper": {
            "paper_id": "2102.00002",
            "title": "Reinforcement Learning for Robotics",
            "abstract": "An RL method for robotic control is introduced and evaluated.",
            "authors": [
              "Alan Turing"
            ],
            "categories": [
              "cs.LG",
              "cs.RO"
            ],
            "published_date": "2021-02-02",
            "venue": null,
            "doi": "10.1234/xyz",
            "arxiv_id": "2102.00002",
            "url": "https://arxiv.org/abs/2102.00002",
            "source": "kaggle_arxiv",
            "citation_count": 0,
            "influential_citation_count": 0,
            "references": [],
            "tldr": "",
            "s2_enriched": false
          },
          "score": 0.66,
          "reason": "Recommended due to high keyword overlap with query terms. Title: \"Reinforcement Learning for Robotics\".",
          "evidence": [
            "2102.00002"
          ],
          "apa_citation": "Turing, A. (2021). Reinforcement Learning for Robotics. *arXiv*. https://doi.org/10.1234/xyz",
          "relation": "similar"
        },
        {
          "paper": {
            "paper_id": "2103.00003",
            "title": "Bayesian Methods in Statistical Machine Learning",
            "abstract": "We study Bayesian inference techniques for ML models.",
            "authors": [
              "Ada Lovelace"
            ],
            "categories": [
              "stat.ML",
              "stat.TH"
            ],
            "published_date": "2021-03-03",
            "venue": null,
            "doi": null,
            "arxiv_id": "2103.00003",
            "url": "https://arxiv.org/abs/2103.00003",
            "source": "kaggle_arxiv",
            "citation_count": 0,
            "influential_citation_count": 0,
            "references": [],
            "tldr": "",
            "s2_enriched": false
          },
          "score": 0.3545,
          "reason": "Recommended due to high semantic similarity between query and abstract. Title: \"Bayesian Methods in Statistical Machine Learning\".",
          "evidence": [
            "2103.00003"
          ],
          "apa_citation": "Lovelace, A. (2021). Bayesian Methods in Statistical Machine Learning. *arXiv*. https://arxiv.org/abs/2103.00003",
          "relation": "similar"
        },
        {
          "paper": {
            "paper_id": "2101.00001",
            "title": "A Transformer Approach to Neural Machine Translation",
            "abstract": "We propose a new transformer model that outperforms prior work on WMT.",
            "authors": [
              "Jane Doe",
              "John Smith"
            ],
            "categories": [
              "cs.CL",
              "cs.LG"
            ],
            "published_date": "2021-01-01",
            "venue": null,
            "doi": null,
            "arxiv_id": "2101.00001",
            "url": "https://arxiv.org/abs/2101.00001",
            "source": "kaggle_arxiv",
            "citation_count": 0,
            "influential_citation_count": 0,
            "references": [],
            "tldr": "",
            "s2_enriched": false
          },
          "score": 0.31,
          "reason": "Recommended due to high semantic similarity between query and abstract. Title: \"A Transformer Approach to Neural Machine Translation\".",
          "evidence": [
            "2101.00001"
          ],
          "apa_citation": "Doe, J., & Smith, J. (2021). A Transformer Approach to Neural Machine Translation. *arXiv*. https://arxiv.org/abs/2101.00001",
          "relation": "similar"
        },
        {
          "paper": {
            "paper_id": "2104.00004",
            "title": "A General Theory of Artificial Intelligence",
            "abstract": "A unifying framework for AI reasoning systems is proposed.",
            "authors": [
              "Geoffrey H"
            ],
            "categories": [
              "cs.AI"
            ],
            "published_date": "2021-04-04",
            "venue": null,
            "doi": null,
            "arxiv_id": "2104.00004",
            "url": "https://arxiv.org/abs/2104.00004",
            "source": "kaggle_arxiv",
            "citation_count": 0,
            "influential_citation_count": 0,
            "references": [],
            "tldr": "",
            "s2_enriched": false
          },
          "score": 0.31,
          "reason": "Recommended due to high semantic similarity between query and abstract. Title: \"A General Theory of Artificial Intelligence\".",
          "evidence": [
            "2104.00004"
          ],
          "apa_citation": "H, G. (2021). A General Theory of Artificial Intelligence. *arXiv*. https://arxiv.org/abs/2104.00004",
          "relation": "similar"
        }
      ],
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
      ]
    }
  },
  "expected_output_contract": {
    "input_contract": "rag_evidence_pack",
    "must_include_source_ids": true,
    "must_separate_findings_from_assumptions": true,
    "must_not_fabricate_metadata": true
  }
}
```

## Output

{
  "summary": "The query focuses on mastering visual continuous control through improved data-augmented reinforcement learning. The retrieved papers cover reinforcement learning for robotics, Bayesian methods in machine learning, transformer models for translation, and a general theory of AI. However, none of the papers directly address visual continuous control or data-augmented reinforcement learning.",
  "key_findings": [
    "An RL method for robotic control is introduced and evaluated [2102.00002].",
    "Bayesian inference techniques for ML models are studied [2103.00003].",
    "A new transformer model outperforms prior work on WMT [2101.00001].",
    "A unifying framework for AI reasoning systems is proposed [2104.00004]."
  ],
  "research_gaps": [
    "None of the retrieved papers directly address visual continuous control or data-augmented reinforcement learning.",
    "The evidence does not provide specific findings on the topic of the query."
  ]
}
