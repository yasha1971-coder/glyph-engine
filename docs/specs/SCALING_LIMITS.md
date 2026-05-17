# GLYPH Scaling Limits

## Current state (v0.x)

GLYPH v0.x uses SA32: suffix array stored as uint32.

Hard format limit:

    corpus ≤ 4,294,967,295 bytes (~4 GiB)

Above this: SA32 overflow → silent corruption.
No overflow detection exists yet. This is a known gap.

---

## RAM economics

Plain FM-index without compression requires approximately:

    SA32:       4 bytes/symbol
    BWT:        1 byte/symbol
    FM index:   ~4 bytes/symbol (checkpoints + Occ)
    corpus:     1 byte/symbol
    ─────────────────────────────
    total:      ~9-10× corpus size in RAM

Observed on benchmark machine (HDFS 1GB):

    corpus:    1.0 GB
    total RAM: ~9.4 GB
    ratio:     9.4×

---

## Practical limits on current machine

Benchmark machine: AMD EPYC 4344P, 118 GB RAM available.

    corpus  5 GB  →  ~47 GB RAM   comfortable
    corpus 10 GB  →  ~94 GB RAM   feasible
    corpus 12 GB  →  ~113 GB RAM  near limit
    corpus  4 GB  →  SA32 format hard ceiling

The binding constraint below 12 GB corpus is SA32 format, not RAM.

---

## Scaling ladder

### Step 1 — SA32 stable path (current)
- works today
- corpus limit: ~4 GiB hard
- RAM limit: ~12 GB corpus on benchmark machine
- status: complete

### Step 2 — SA64 path (next)
- suffix array stored as uint64
- removes 4 GiB hard ceiling
- unlocks corpus 4–12 GB on current machine
- same RAM ratio (~9.4×)
- requires: builder changes, query binary changes, format versioning
- status: designed in SA64_DESIGN.md, not yet implemented

### Step 3 — Segmented SA64
- multiple SA64 shards
- fan-out query across shards
- unlocks corpus beyond single machine RAM
- cross-shard boundary matches still not detected
- status: planned

### Step 4 — Compressed / sampled SA
- sampled suffix array (every k-th entry stored)
- wavelet tree for Occ table
- reduces RAM ratio from ~9.4× to ~2-3×
- unlocks corpus 50–100+ GB on reasonable hardware
- locate cost increases by O(k) LF steps
- status: research

---

## Why SA64 before compressed SA

Compressed SA is more complex and changes the correctness model.
SA64 is a format change with the same algorithm.

SA64 on current machine unlocks:
- corpus up to ~12 GB (RAM bound)
- multi-shard corpus beyond 4 GB per shard

Compressed SA becomes necessary only when:
- corpus exceeds available RAM / 9.4
- or RAM economics become the primary constraint

On a 118 GB machine, that threshold is ~12 GB corpus.
Below that, SA64 is sufficient.

---

## Known gaps before SA64

Before implementing SA64:

- [ ] SA32 overflow detection at build time (hard error, not silent)
- [ ] index format versioning (magic bytes + version field)
- [ ] corpus hash in index header (stale index detection)

These must exist in SA32 path first.
SA64 inherits the same format discipline.

---

## Summary

| Limit | Value | Cause |
|---|---|---|
| SA32 corpus ceiling | ~4 GiB | uint32 overflow |
| RAM practical ceiling | ~12 GB corpus | 9.4× ratio × 118 GB |
| Next unlock | SA64 path | format change only |
| Long-term unlock | compressed SA | algorithmic change |
