# Dataset Limitations

### 1. Scope is CS/ML-only, not broad STEM
The corpus is restricted to `cs.CL`, `cs.AI`, `cs.LG`, and `stat.ML`.
Papers in physics, biology, chemistry, or engineering are excluded.
This makes the recommendation system domain-specific; cross-domain
retrieval is out of scope for the current system.

### 2. Abstracts only — no full text
The Kaggle arXiv dump provides only titles, abstracts, and metadata.
Full paper bodies (methods, results, figures, tables) are unavailable.
This limits extractive summarisation quality and means the system cannot
reason about experimental details or figure content.
*(Multimodal figure/table understanding is noted as a planned future extension.)*

### 3. Sparse metadata
A significant proportion of records lack DOI and venue fields, as the raw
Kaggle dump reflects the arXiv pre-print state of papers (many never
formally published, or journal-ref not updated). Semantic Scholar
enrichment partially mitigates this but is not exhaustive.

### 4. Label imbalance
The primary-category distribution is uneven (cs.LG > cs.CL > cs.AI >
stat.ML in most time windows). Classifiers trained without rebalancing
will be biased toward cs.LG. We address this with class-weighted loss
in Stages 05–06.

### 5. Temporal bias
The subset is sampled in file order, which skews toward older papers
(the Kaggle dump is roughly chronological). Recent papers (2023–2024)
are underrepresented relative to their actual volume in arXiv.
A production system should use stratified temporal sampling.

### 6. Citation lag
Citation counts from Semantic Scholar reflect the state at enrichment
time. Newly published papers will have artificially low counts,
depressing their recommendation ranking. Time-normalised citation
scores (citations per year) would partially correct this.

### 7. No author disambiguation
Author names are taken as-is from the structured `authors_parsed` field.
Name collisions (same name, different researcher) and variants (initials
vs full name) are not resolved. This affects the prolific-author
statistics and citation-graph construction.
