# GLYPH Query Tiers

## Overview

GLYPH has three explicit query tiers.
Each tier adds guarantees and costs.
Tiers are not interchangeable — choose based on use case.

---

## Tier 1 — Fast count

Tool: persistent FM backend (`query_fm_server_v1`)

What it does:
- FM backward search only
- returns count = r - l from FM interval

What it does NOT do:
- no manifest verification
- no corpus integrity check
- no offset recovery
- no locate

Latency (HDFS 1GB, warm):
    p50: ~0.010 ms
    p99: ~0.015 ms

When to use:
- repeated exact queries over a trusted static corpus
- when count is sufficient
- when index is known-good and corpus is unchanged

Risk:
- manifest verification depends on the correctness of manifest.json
- locate is not included
- artifact checksum is not embedded inside fm.bin yet

---

## Tier 2 — Verified query

Tool: `tools/query_verified_v1.py`

What it does:
- manifest verification before query
- corpus sha256 check
- sentinel value check
- artifact existence check
- FM count query
- fail-fast on any mismatch

What it does NOT do:
- no locate (offset recovery)
- no per-query artifact checksum (manifest check only)

Latency (HDFS 1GB, warm):
    ~19 ms end-to-end
    (Python startup + manifest verification + subprocess)

When to use:
- CLI use where integrity matters
- when corpus may have changed between queries
- when artifact provenance must be confirmed before result

Risk:
- manifest check uses sha256 of corpus prefix (64KB)
- not a full corpus hash on every query
- locate not included

---

## Tier 3 — Strict verified (planned)

Status: not yet implemented.

Intended behavior:
- all Tier 2 checks
- full corpus sha256 (not prefix only)
- artifact checksum inside FM binary header
- verified locate: count == len(offsets), corpus[o:o+len(p)] == p
- explicit NotFound signal distinct from count=0

When to use:
- provenance audit
- forensic / compliance use
- when exact byte offsets must be verified against corpus

Latency: not yet measured. Expected higher than Tier 2.

---

## Tier comparison

| Property | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| FM count | ✓ | ✓ | ✓ |
| Manifest check | — | ✓ | ✓ |
| Full corpus hash | — | — | ✓ |
| Artifact checksum | — | — | ✓ |
| Locate + verify | — | — | ✓ |
| Explicit NotFound | — | — | ✓ |
| Latency | ~0.010 ms | ~19 ms | TBD |

---

## Design principle

Each tier must fail-fast on its own guarantees.
No tier silently falls back to a weaker tier.
A Tier 2 query that fails manifest check must not proceed to FM count.
A Tier 3 query that fails artifact checksum must not proceed to locate.

---

## Relationship to artifact protocol

Tier 1 → requires valid FM artifact (magic + version)
Tier 2 → requires valid manifest.json + corpus
Tier 3 → requires Tier 2 + artifact checksum inside fm.bin
