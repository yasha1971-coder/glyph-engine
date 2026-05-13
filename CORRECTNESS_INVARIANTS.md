# CORRECTNESS INVARIANTS

This document defines the current correctness contracts of GLYPH v0.x.

The goal is to formalize:
- retrieval semantics
- index assumptions
- deterministic guarantees
- sentinel behavior
- component contracts

These invariants are treated as architectural constraints, not implementation details.

---

# 1. Corpus invariants

## 1.1 Static corpus assumption

GLYPH v0.x assumes:
- immutable corpora
- offline index construction
- read-only query behavior

Indexes are not currently designed for dynamic mutation.

---

## 1.2 Raw byte semantics

GLYPH operates over raw bytes.

No:
- tokenization
- normalization
- encoding interpretation
- semantic preprocessing

All retrieval semantics are byte-exact.

---

## 1.3 Sentinel exclusion invariant

Input corpus MUST NOT contain:

0x00

Reason:
GLYPH v0.x appends a real terminal sentinel byte during index construction.

Violation of this invariant invalidates retrieval semantics.

Future versions may replace this with:
- 257-symbol alphabets
- out-of-band sentinel representations
- alternative suffix termination schemes

---

# 2. Sentinel invariants

## 2.1 Real appended sentinel

GLYPH v0.x indexes:

raw_corpus + appended_terminal_0x00

The sentinel is physically appended to indexed corpus data.

This is NOT a synthetic virtual sentinel.

---

## 2.2 Exactly one sentinel

Indexed corpus MUST contain exactly one terminal sentinel.

Invariant:

count(0x00) == 1

---

## 2.3 Sentinel query behavior

Querying:

b"\x00"

MUST return exactly one match.

This behavior is intentional and currently part of retrieval semantics.

---

# 3. Suffix array invariants

## 3.1 SA correctness

Suffix array ordering MUST preserve lexicographic suffix ordering over the sentinel-augmented corpus.

---

## 3.2 SA determinism

Given identical corpus bytes:
- suffix array construction MUST produce identical output
- retrieval intervals MUST remain stable

---

## 3.3 SA ↔ corpus consistency

All suffix array offsets MUST reference valid corpus positions.

No offset may exceed corpus length.

---

# 4. BWT invariants

## 4.1 True BWT construction

BWT MUST be constructed from:
- the real suffix array
- the real sentinel-appended corpus

Synthetic sentinel substitution is forbidden.

---

## 4.2 BWT length invariant

Invariant:

len(BWT) == len(corpus_with_sentinel)

---

## 4.3 Sentinel consistency

The sentinel position implied by:
- corpus
- suffix array
- BWT
- FM-index

MUST remain globally consistent.

---

# 5. FM-index invariants

## 5.1 FM interval determinism

Identical:
- corpus
- query
- index files

MUST produce identical intervals.

---

## 5.2 Exact occurrence counting

FM counts MUST match exact substring occurrence counts over:
- overlapping matches
- single-byte patterns
- full-corpus matches

Python byte-search oracle currently acts as verification reference.

---

## 5.3 No probabilistic retrieval

FM retrieval semantics are exact.

No:
- approximation
- fuzzy matching
- ranking
- heuristic expansion

---

# 6. Query invariants

## 6.1 Exact byte matching

Queries are matched byte-for-byte.

No:
- Unicode normalization
- stemming
- lowercasing
- tokenizer behavior

---

## 6.2 Overlapping occurrence semantics

Overlapping matches MUST be counted.

Example:

corpus = b"aaaa"
pattern = b"aa"

Expected count:

3

Offsets:

[0,1,2]

---

## 6.3 Deterministic query semantics

Repeated identical queries over identical indexes MUST return identical results.

---

# 7. Retrieval semantics

GLYPH currently guarantees only:

- exact byte-level presence
- deterministic retrieval behavior
- stable retrieval semantics
- exact occurrence counting

GLYPH does NOT guarantee:
- semantic correctness
- factual correctness
- relevance
- ranking quality
- interpretation quality

---

# 8. Current verification mechanisms

Current verification includes:
- regression correctness tests
- Python oracle comparison
- sentinel-specific tests
- deterministic repeated-query checks
- HDFS correctness validation

---

# 9. Known limitations

Current limitations include:
- high RAM overhead
- static corpus assumption
- evolving APIs
- incomplete locate-layer verification
- incomplete large-scale fuzz coverage

GLYPH v0.x should currently be treated as experimental infrastructure research.

---

# Core principle

Correctness is treated as an architectural property.

Not:
- a benchmark optimization
- a side effect
- a probabilistic approximation
- a UI feature
