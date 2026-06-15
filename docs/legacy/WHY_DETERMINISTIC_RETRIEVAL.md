# WHY DETERMINISTIC RETRIEVAL

Modern retrieval systems increasingly optimize for probabilistic relevance.

This is often useful:
- semantic search
- embeddings
- reranking
- contextual retrieval
- approximate nearest-neighbor systems

However, these systems also introduce uncertainty.

The same query may:
- return different results over time
- depend on ranking heuristics
- depend on model updates
- depend on embedding drift
- lose exact byte provenance

GLYPH explores the opposite direction.

---

## Core idea

GLYPH treats retrieval as an exact infrastructure problem.

Goal:

    same bytes in → same matches out

The system operates over:
- static corpora
- exact byte substrings
- deterministic index structures

No semantic interpretation is required.

---

## Why exactness matters

Exact retrieval becomes important when systems need:

- reproducibility
- auditability
- exact provenance
- stable byte offsets
- deterministic verification
- low-level observability

Examples include:
- infrastructure logs
- binary corpora
- forensic analysis
- retrieval validation
- exact post-filtering beneath probabilistic systems

---

## Deterministic vs probabilistic retrieval

Probabilistic systems are often optimized for:
- usefulness
- semantic flexibility
- approximate intent matching

Deterministic systems optimize for:
- exact presence
- reproducibility
- stable retrieval semantics
- infrastructure predictability

These goals are different.

GLYPH does not attempt to replace probabilistic systems.

Instead, it explores whether exact deterministic retrieval can serve as a stable verification substrate beneath them.

---

## Exact verification layer

One possible future architecture:

LLM
↓
semantic retrieval
↓
reranker
↓
GLYPH exact verifier
↓
exact byte offsets
↓
ground-truth confirmation

In this model:
- probabilistic systems generate candidates
- deterministic systems verify exact presence

---

## Current limitations

GLYPH is currently experimental.

Known limitations include:
- high RAM overhead
- static-corpus assumptions
- evolving APIs
- incomplete correctness coverage
- limited operational hardening

The project is infrastructure research, not a production platform.

---

## Research direction

GLYPH currently explores:
- FM-index infrastructure
- suffix-array retrieval
- deterministic substring search
- mmap-based retrieval behavior
- exact byte-offset reproducibility
- retrieval observability
- static-corpus verification semantics

Core principle:

    exactness is a capability,
    not a byproduct.
