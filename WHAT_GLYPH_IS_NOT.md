# WHAT GLYPH IS NOT

GLYPH is frequently misunderstood because most modern retrieval systems
optimize for semantic flexibility rather than deterministic exactness.

This document clarifies what GLYPH does NOT attempt to be.

---

## GLYPH is NOT a search engine

GLYPH does not provide:
- ranking
- fuzzy matching
- typo correction
- semantic interpretation
- relevance scoring
- query understanding

Same bytes in → same matches out.

---

## GLYPH is NOT a vector database

GLYPH does not:
- store embeddings
- compute similarity
- perform nearest-neighbor search
- approximate meaning

It operates on exact byte substrings over static corpora.

---

## GLYPH is NOT an LLM framework

GLYPH does not:
- orchestrate agents
- manage prompts
- execute tools
- generate text
- perform reasoning

GLYPH is infrastructure-level retrieval software.

---

## GLYPH is NOT a replacement for grep

For one-off scans, grep is often simpler and cheaper.

GLYPH trades:
- RAM
- preprocessing
- offline indexing

for:
- deterministic repeated queries
- low-latency retrieval
- stable exact-match behavior over static corpora

---

## GLYPH is NOT currently production-ready

Current limitations include:
- high RAM overhead
- evolving APIs
- experimental architecture
- incomplete correctness coverage
- limited operational hardening

The project is currently an experimental infrastructure prototype.

---

## GLYPH does NOT claim patent safety

GLYPH uses public algorithms and techniques.

No patent clearance or legal safety guarantee is claimed.

Commercial usage requires independent legal review.

---

## What GLYPH actually is

GLYPH is:

- deterministic byte-exact retrieval
- suffix-array / BWT / FM-index infrastructure
- exact substring retrieval over static corpora
- reproducible retrieval behavior
- a possible exact-verification layer beneath probabilistic systems

Core idea:

    probabilistic systems find candidates;
    deterministic systems verify exact presence.
