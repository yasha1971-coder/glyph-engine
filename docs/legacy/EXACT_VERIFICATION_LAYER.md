# EXACT VERIFICATION LAYER

Modern retrieval systems increasingly rely on probabilistic pipelines.

Typical stacks now include:
- embeddings
- semantic retrieval
- reranking
- contextual chunking
- agent orchestration
- adaptive retrieval loops

These systems are often powerful and useful.

However, they also introduce uncertainty.

Examples:
- approximate matches
- embedding drift
- ranking instability
- changing retrieval semantics
- inconsistent provenance
- retrieval non-reproducibility

GLYPH explores a complementary direction.

---

## Core idea

GLYPH investigates whether deterministic exact retrieval can function as a stable verification substrate beneath probabilistic systems.

Instead of replacing semantic retrieval, GLYPH focuses on:

- exact byte presence
- deterministic retrieval behavior
- stable byte offsets
- reproducible retrieval semantics
- exact provenance anchors

Goal:

    probabilistic systems retrieve candidates;
    deterministic systems verify exact presence.

---

## Verification vs interpretation

Probabilistic systems optimize for:
- semantic usefulness
- approximate intent matching
- contextual relevance

Verification systems optimize for:
- exact presence
- reproducibility
- deterministic observability
- stable references

These are different infrastructure roles.

GLYPH focuses on the second role.

---

## Possible retrieval architecture

One possible future pipeline:

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
- semantic systems generate candidate regions
- GLYPH verifies exact byte-level existence

---

## Why this may matter

As retrieval systems become more probabilistic, infrastructure may require stronger deterministic anchors.

Examples:
- audit systems
- infrastructure observability
- legal/compliance workflows
- forensic analysis
- binary corpus verification
- reproducible AI retrieval pipelines
- exact provenance tracking

GLYPH explores whether exact deterministic retrieval can provide such anchors.

---

## Important boundaries

GLYPH does NOT:
- prove semantic truth
- validate reasoning
- solve hallucinations
- guarantee factual correctness
- replace semantic retrieval systems

GLYPH only verifies exact byte-level presence within indexed static corpora.

---

## Current research areas

Current exploration includes:
- FM-index infrastructure
- suffix-array retrieval
- mmap retrieval behavior
- deterministic substring search
- exact byte-offset recovery
- retrieval reproducibility
- static-corpus verification semantics
- sentinel-safe index construction

---

## Experimental status

GLYPH is currently experimental infrastructure research.

Known limitations include:
- high RAM overhead
- evolving APIs
- incomplete correctness coverage
- limited operational hardening
- static-corpus assumptions

The project should not currently be treated as production infrastructure.

---

## Core principle

GLYPH explores a simple question:

    can exact deterministic retrieval remain stable
    beneath increasingly probabilistic systems?
