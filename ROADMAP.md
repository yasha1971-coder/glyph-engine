# GLYPH Roadmap

## v0.1 — Current

Current state:

- exact byte-level retrieval
- SA32u
- BWT
- FM-index
- persistent FM query backend
- 1GB / 4GB validated corpora
- HDFS benchmark published

Known tradeoffs:

- very high RAM overhead
- static corpora only
- no regex / ranking / fuzzy search
- no segmented retrieval yet

---

## v0.2 — Segmented Retrieval

Goal:

Break the 4GB single-shard ceiling.

Planned work:

- segmented corpus layout
- shard routing
- merged shortlist retrieval
- shard-local SA/BWT/FM
- deterministic cross-shard query merge

Target:

- 50GB+ static corpora
- bounded RAM per shard

---

## v0.3 — RAM Reduction

Goal:

Reduce persistent memory overhead.

Research directions:

- mmap-based access
- compressed FM structures
- sampled SA
- lazy loading
- MADV_RANDOM / paging experiments

Target:

- lower RAM amplification
- larger corpora on commodity hardware

---

## Non-goals

GLYPH is not:

- a search relevance engine
- a vector database
- a replacement for Elasticsearch
- a fuzzy matcher
- a regex engine

GLYPH explores a different tradeoff:

persistent byte-level indexed retrieval over static corpora.
