# P04 citation_recommendation

- Backend: `ollama`
- Model: `qwen3:8b`
- Strategy: `zero_shot`
- Latency seconds: `67.7123`
- Error: `none`
- Evidence IDs used: `2102.00002`

## Prompt

```text
You are a local research-paper assistant. Answer only from ParsedPaper fields, RagEvidencePack evidence snippets, Recommendation metadata, or explicitly stated assumptions. Cite source IDs for evidence-backed claims. Do not invent authors, venues, citations, scores, or findings. If evidence is incomplete, say so.

Style: technical.

Task: recommend citations with APA-style strings from Recommendation metadata.
Never fabricate missing DOI, venue, year, or author details.

Return the final user-visible answer only. Use compact Markdown sections.

Input JSON:

{
  "task": "citation_recommendation",
  "style": "technical",
  "input": {
    "parsed_paper": {
      "metadata": {
        "paper_id": "2107.09645v1_fd56aa5c6a",
        "title": "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning",
        "abstract": "We present DrQ-v2, a model-free reinforcement learning (RL) algorithm for visual continuous control. DrQ-v2 builds on DrQ, an off-policy actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements that yield state-of-the-art results on the DeepMind Control Suite. Notably, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL. DrQ-v2 is conceptually simple, easy to implement, and provides signi\ufb01cantly better computational footprint compared to prior work, with the majority of tasks taking just 8 hours to train on a single GPU. Finally, we publicly release DrQ-v2\u2019s implementation to provide RL practitioners with a strong and computationally ef\ufb01cient baseline.",
        "authors": [
          "Denis Yarats",
          "Rob Fergus",
          "Alessandro Lazaric",
          "Lerrel Pinto"
        ],
        "categories": [],
        "published_date": null,
        "venue": null,
        "doi": null,
        "arxiv_id": "2107.09645v1",
        "url": "https://arxiv.org/abs/2107.09645v1",
        "source": "uploaded_pdf"
      },
      "sections": {
        "abstract": "We present DrQ-v2, a model-free reinforcement learning (RL) algorithm for visual continuous control. DrQ-v2 builds on DrQ, an off-policy actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements that yield state-of-the-art results on the DeepMind Control Suite. Notably, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL. DrQ-v2 is conceptually simple, easy to implement, and provides signi\ufb01cantly better computational footprint compared to prior work, with the majority of tasks taking just 8 hours to train on a single GPU. Finally, we publicly release DrQ-v2\u2019s implementation to provide RL practitioners with a strong and computationally ef\ufb01cient baseline.",
        "introduction": "Creating sample-ef\ufb01cient continuous control methods that observe high-dimensional images has\nbeen a long standing challenge in reinforcement learning (RL) . Over the last three years, the RL\ncommunity has made signi\ufb01cant headway on this problem, improving sample-ef\ufb01ciency signi\ufb01cantly.\nThe key insight to solving visual control is the learning of better low-dimensional representations,\neither through autoencoders [Yarats et al., 2019, Finn et al., 2015], variational inference [Hafner\net al., 2018, 2019, Lee et al., 2019], contrastive learning [Srinivas et al., 2020, Yarats et al., 2021a],\nself-prediction [Schwarzer et al., 2020b], or data augmentations [Yarats et al., 2021b, Laskin et al.,\n2020]. However, current state-of-the-art model-free methods are still limited in three ways. First, they\nare unable to solve the more challenging visual control problems such as quadruped and humanoid\nlocomotion. Second, they often require signi\ufb01cant computational resources, i.e. lengthy training\ntimes using distributed multi-GPU infrastructure. Lastly, it is often unclear how different design\nchoices affect overall system performance.\nIn this paper we present DrQ-v2, a simple model-free algorithm that builds on the idea of using\ndata augmentations [Yarats et al., 2021b, Laskin et al., 2020] to solve hard visual control problems.\nMost notably, it is the \ufb01rst model-free method that solves complex humanoid tasks directly from\npixels. Compared to previous state-of-the-art model-free methods, DrQ-v2 provides signi\ufb01cant\nimprovements in sample ef\ufb01ciency across tasks from the DeepMind Control Suite [Tassa et al., 2018].\nConceptually simple, DrQ-v2 is also computationally ef\ufb01cient, which allows solving most tasks in\nDeepMind Control Suite in just 8 hours on a single GPU (see Figure 1). Recently, a model-based\nmethod, DreamerV2 [Hafner et al., 2020] was also shown to solve visual continuous control problems\nand it was \ufb01rst to solve the humanoid locomotion problem from pixels. While our model-free\nDrQ-v2 matches DreamerV2 in terms sample ef\ufb01ciency and performance, it does so 4\u00d7 faster in\nterms of wall-clock time to train. We believe this makes DrQ-v2 a more accessible approach to\nsupport research in visual continuous control and it reinforces the question on whether model-free or\nmodel-based is the more suitable approach to solve this type of tasks.\nDrQ-v2, which is detailed in Section 3, improves upon DrQ [Yarats et al., 2021b] by making\nseveral algorithmic changes: (i) switching the base RL algorithm from SAC [Haarnoja et al., 2018b]\nto DDPG [Lillicrap et al., 2015a], (ii) this allows us straightforwardly incorporating multi-step\nreturn, (iii) adding bilinear interpolation to the random shift image augmentation, (iv) introducing an\nexploration schedule, (v) selecting better hyper-parameters including a larger capacity of the replay\nbuffer. A careful ablation study of these design choices is presented in Section 4.4. Furthermore, we\nre-examine the original implementation of DrQ and identify several computational bottlenecks such\n2\nas replay buffer management, data augmentation processing, batch size, and frequency of learning\nupdates (see Section 3.2). To remedy these, we have developed a new implementation that both\nachieves better performance and trains around 3.5 times faster with respect to wall-clock time than\nthe previous implementation on the same hardware with an increase in environment frame throughput\n(FPS) from 28 to 96 (i.e., it takes 106/96/3600 \u2248 2.9 hours to train for 1M environment steps).",
        "method": "Image Augmentation As in DrQ we apply random shifts image augmentation to pixel observations\nof the environment. In the settings of visual continuous control by DMC, this augmentation can be\n1Here, \u2206(X ) denotes a distribution over the state space X .\n\nInputs:\nf\u03be,\u03c0\u03c6,Q\u03b81,Q\u03b82: parametric networks for encoder, policy, and Q-functions respectively.\naug: random shifts image augmentation.\n\u03c3(t): scheduled standard deviation for the exploration noise de\ufb01ned in Equation (3).\nT ,B,\u03b1,\u03c4,c: training steps, mini-batch size, learning rate, target update rate, clip value.\nTraining routine:\nfor each timestept = 1..T do\n\u03c3t \u2190\u03c3(t) \u22bf Compute stddev for the exploration noise\nat \u2190\u03c0\u03c6(f\u03be(xt)) +\u03f5 and\u03f5 \u223c N (0,\u03c3 2\nt ) \u22bf Add noise to the deterministic action\nxt+1 \u223cP (\u00b7|xt, at) \u22bf Run transition function for one step\nD \u2190 D \u222a (xt, at,R (xt, at), xt+1) \u22bf Add a transition to the replay buffer\nUPDATE CRITIC (D,\u03c3t)\nUPDATE ACTOR (D,\u03c3t)\nend for\nprocedure UPDATE CRITIC (D,\u03c3 )\n{(xt, at,rt:t+n\u22121, xt+n)} \u223c D \u22bf Sample a mini batch ofB transitions\nht, ht+n \u2190f\u03be(aug(xt)),f\u03be(aug(xt+n)) \u22bf Apply data augmentation and encode\nat+n \u2190\u03c0\u03c6(ht+n) +\u03f5 and\u03f5 \u223c clip(N (0,\u03c3 2)) \u22bf Sample action\nCompute L\u03b81,\u03be and L\u03b82,\u03be using Equation (1) \u22bf Compute critic losses\n\u03be \u2190\u03be \u2212\u03b1\u2207\u03be(L\u03b81,\u03be + L\u03b82,\u03be) \u22bf Update encoder weights\n\u03b8k \u2190\u03b8k \u2212\u03b1\u2207\u03b8k L\u03b8k,\u03be \u2200k \u2208 {1, 2} \u22bf Update critic weights\n\u00af\u03b8k \u2190 (1 \u2212\u03c4)\u00af\u03b8k +\u03c4\u03b8k \u2200k \u2208 {1, 2} \u22bf Update critic target weights\nend procedure\nprocedure UPDATE ACTOR (D,\u03c3 )\n{(xt)} \u223c D \u22bf Sample a mini batch ofB observations\nht \u2190f\u03be(aug(xt)) \u22bf Apply data augmentation and encode\nat \u2190\u03c0\u03c6(ht) +\u03f5 and\u03f5 \u223c clip(N (0,\u03c3 2)) \u22bf Sample action\nCompute L\u03c6 using Equation (2) \u22bf Compute actor loss\n\u03c6 \u2190\u03c6 \u2212\u03b1\u2207\u03c6L\u03c6 \u22bf Update actor\u2019s weights only\nend procedure\ninstantiated by \ufb01rst padding each side of 84 \u00d7 84 observation rendering by 4 pixels (by repeating\nboundary pixels), and then selecting a random 84 \u00d7 84 crop, yielding the original image shifted by\n\u00b14 pixels. We also \ufb01nd it useful to apply bilinear interpolation on top of the shifted image (i.e, we\nreplace each pixel value with the average of the four nearest pixel values). In our experience, this\nmodi\ufb01cation provides an additional performance boost across the board.\nImage Encoder The augmented image observation is then embedded into a low-dimensional latent\nvector by applying a convolutional encoder. We use the same encoder architecture as in DrQ, which\n\ufb01rst was introduced introduced in SAC-AE [Yarats et al., 2019]. This process can be succinctly\nsummarized as h =f\u03be(aug(x)), wheref\u03be is the encoder, aug is the random shifts augmentation,\nand x is the original image observation.\nActor-Critic Algorithm We use DDPG [Lillicrap et al., 2015a] as a backbone actor-critic RL\nalgorithm and, similarly to Barth-Maron et al. [2018], augment it withn-step returns to estimate TD\nerror. This results into faster reward propagation and overall learning progress [Mnih et al., 2016a].\nWhile some methods [Hafner et al., 2020] employ more sophisticated techniques such as TD(\u03bb) or\nRetrace(\u03bb) [Munos et al., 2016], they are often computationally demanding whenn is large. We \ufb01nd\nthat using simplen-step returns, without an importance sampling correction, strikes a good balance\nbetween performance and ef\ufb01ciency. We also employ clipped double Q-learning [Fujimoto et al.,\n2018] to reduce overestimation bias in the target value. Practically, this requires training two Q-\nfunctionsQ\u03b81 andQ\u03b82. For this, we sample a mini-batch of transitions\u03c4 = (xt, at,rt:t+n\u22121, xt+n)\nfrom the replay buffer D and compute the following two losses:\nL\u03b8k,\u03be(D) = E\u03c4\u223cD\n[\n(Q\u03b8k(ht, at) \u2212y)2]\n\u2200k \u2208 {1, 2}, (1)\n4\nwith the TD targety de\ufb01ned as:\ny =\nn\u22121\u2211\ni=0\n\u03b3irt+i +\u03b3n min\nk=1,2\nQ\u00af\u03b8k(ht+n, at+n),\nwhere ht =f\u03be(aug(xt)), ht+n =f\u03be(aug(xt+n)), at+n =\u03c0\u03c6(ht+n) +\u03f5, and \u00af\u03b81,\u00af\u03b82 are the slow-\nmoving weights for the Q target networks. We note, that in contrast to DrQ, we do not employ a\ntarget network for the encoderf\u03be and always use the most recent weights\u03be to embed xt and xt+n.\nThe exploration noise\u03f5 is sampled from clip(N (0,\u03c3 2), \u2212c,c ) [\u2026section truncated for summarization\u2026]",
        "results": "In this section we provide empirical evaluation of DrQ-v2 on an extensive set of visual continuous\ncontrol tasks from DMC [Tassa et al., 2018]. We \ufb01rst present comparison to prior methods, both\nmodel-free and model-based, in terms of sample ef\ufb01ciency and wall-clock time. We then present a\nlarge scale ablation study that guided the \ufb01nal version of DrQ-v2.\n5\n0 5 10 15 20 25 30\nFrames (\u00d7106)\n0\n200\n400\n600\n800Episode Return\nHumanoid Stand\nSAC\nCURL\nDrQ\nDrQ-v2\n0 5 10 15 20 25 30\nFrames (\u00d7106)\n0\n200\n400\n600\n800Episode Return\nHumanoid Walk\n0 5 10 15 20 25 30\nFrames (\u00d7106)\n0\n50\n100\n150\n200\n250Episode Return\nHumanoid Run\nFigure 3: The hard benchmark consists of three humanoid locomotion tasks: stand, walk, and\nrun. These three represent particularly hard exploration challenges, being previously unsolvable by\nmodel-free methods. The training speed of DrQ-v2 was key to solving the task, since it allowed for\nextensive investigation of different variations, resulting in the discovery of effective strategies.",
        "conclusion": "We have introduced a conceptually simple model-free actor-critic RL algorithm for image-based\ncontinuous control \u2013 DrQ-v2. Our method provides signi\ufb01cantly better computational footprint\nand masters tasks from DMC [Tassa et al., 2018] directly from pixels, most notably the humanoid\nlocomotion tasks that were previously unsolved by model-free approaches. Additionally, we have\nprovided an ef\ufb01cient PyTorch implementation of DrQ-v2 that is publicly available at https://\ngithub.com/facebookresearch/drqv2. We hope that our algorithm will help to inspire and\ndemocratize further research in visual RL.",
        "related_work": "Visual Reinforcement Learning Successes of visual representation learning in computer vi-\nsion [Vincent et al., 2008, Doersch et al., 2015, Wang and Gupta, 2015, Noroozi and Favaro, 2016,\nZhang et al., 2017, Gidaris et al., 2018] has inspired successes in visual RL, where coherent repre-\nsentations are learned alongside RL. Works such as SAC-AE [Yarats et al., 2019], PlaNet [Hafner\net al., 2018], and SLAC [Lee et al., 2019], demonstrated how auto-encoders [Finn et al., 2015] could\nimprove visual RL. Following this, other self-supervised objectives such as contrastive learning\nin CURL [Srinivas et al., 2020] and ATC [Stooke et al., 2020], self-prediction in SPR [Schwarzer\net al., 2020a], contrastive cluster assignment in Proto-RL [Yarats et al., 2021a], and augmented\ndata in DrQ [Yarats et al., 2021b] and RAD [Laskin et al., 2020], have signi\ufb01cantly bridged the\n10\n0 2 4 6 8 10\nHours (for 1 \u00d7 106 Frames)\n0\n200\n400\n600\n800Episode Return\n2.9 hours\nCartpole Swingup\nDreamer-v2\nDrQ-v2\n0 2 4 6 8 10\nHours (for 1 \u00d7 106 Frames)\n0\n200\n400\n600\n800Episode Return 2.9 hours\nFinger Spin\n0 2 4 6 8 10\nHours (for 1 \u00d7 106 Frames)\n0\n200\n400\n600\n800\n1000Episode Return\n2.9 hours\nWalker Stand\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600Episode Return\n8.6 hours\nAcrobot Swingup\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600\n800Episode Return\n8.6 hours\nCheetah Run\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600\n800\n1000Episode Return\n8.6 hours\nFinger Turn Hard\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n100\n200\n300\n400\n500Episode Return\n8.6 hours\nHopper Hop\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600\n800\n1000Episode Return\n8.6 hours\nQuadruped Walk\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600\n800\n1000Episode Return\n8.6 hours\nQuadruped Run\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600\n800\n1000Episode Return 8.6 hours\nReacher Hard\n0 10 20 30\nHours (for 3 \u00d7 106 Frames)\n0\n200\n400\n600\n800Episode Return 8.6 hours\nWalker Run\n0 100 200 300\nHours (for 30 \u00d7 106 Frames)\n0\n200\n400\n600\n800Episode Return 86 hours\nHumanoid Walk\nFigure 8: Model-based Dreamer-v2 performs more computations than model-free DrQ-v2. This\nallows DrQ-v2 to train faster in terms of wall-clock time and outperform Dreamer-v2 in this aspect.\ngap between state-based and image-based RL. Future prediction objectives [Hafner et al., 2018,\n2019, Yan et al., 2020, Finn et al., 2015, Pinto et al., 2016, Agrawal et al., 2016] and other auxiliary\nobjectives [Jaderberg et al., 2016, Zhan et al., 2020, Young et al., 2020, Chen et al., 2020] have shown\nimprovements on a variety of problems ranging from gameplay, continuous control, and robotics. In\nthe context of visual control settings, clever use of augmented data [Yarats et al., 2021b, Laskin et al.,\n2020] currently produces state-of-the-art results on visual tasks from DMC [Tassa et al., 2018].\nHumanoid Control The humanoid control problem \ufb01rst presented in Tassa et al. [2012], has been\nstudied as one of the hardest control problems due to its large state and action spaces. The earliest\nsolutions to this problem use ideas in model-based optimal control to generate policies given an\naccurate model of the humanoid . Subsequent works in RL have shown that model-free policies can\nsolve the humanoid control problem given access to proprioceptive state observations. However,\nsolving such a problem from visual observations has been a challenging problem, with leading RL\nalgorithms making little progress to solve the task [Tassa et al., 2018]. Recently, Hafner et al. [2020]\nwas able to solve this problem through a model-based technique in around 30M environment steps\nand 340 hours of training on a single GPU machine. DrQ-v2, presented in this paper, marks the \ufb01rst\nmodel-free RL method that can solve humanoid control from visual observations, taking also around\n30M steps and 86 hours of training on the same hardware.\n11\n0.0 0.5 1.0 1.5 2.0 2.5 3.0\nFrames (\u00d7106)\n0\n50\n100\n150\n200Episode Return\nAcrobot Swingup\nDrQ:SAC\nDrQ:DDPG\n0.0 0.5  [\u2026section truncated for summarization\u2026]"
      },
      "keywords": [],
      "entities": {
        "methods": [],
        "datasets": [],
        "tasks": [],
        "metrics": [],
        "institutions": []
      }
    },
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
