# GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1

Status: executable conformance closure
Version: 1
Date: 2026-07-13

## Purpose

Bind the verified GLYPH reference semantics P1-P12 to the actual compiled
binary-safe C++ runtime.

The reference proof graph defines correctness.

The runtime conformance graph proves that the real implementation preserves
that correctness through count, locate, multi-document search, evidence replay,
and a self-contained portable bundle.

## External dependency

The runtime graph requires:

    GLYPH_PROOF_GRAPH_V1
    P1 through P12 PASS
    GLYPH PROOF GRAPH OK

## Runtime nodes

### R0 — Legacy Runtime Baseline

Confirms the historical sentinel-safe v0.x baseline:

- source 0x00 rejected;
- physical 0x00 used as sentinel;
- uint8 BWT;
- 256-symbol FM alphabet;
- runtime conformant false.

R0 is an audit baseline, not a conformance success.

### R1 — Binary-safe C++ Count

Requires:

- source byte domain 0x00 through 0xFF;
- virtual sentinel 256;
- n+1 suffix-array rows;
- uint16 BWT symbols;
- 257-symbol FM alphabet;
- exact C++ count against the byte oracle.

### R2 — Binary-safe C++ Locate

Requires:

- exact FM interval;
- exact canonical offsets;
- bounded locate;
- byte verification;
- terminal suffix never returned as a byte match.

### R3 — Multi-document Runtime

Requires:

- one independent index per document;
- canonical `(document_id, document_offset)` coordinates;
- preservation of empty and duplicate documents;
- no physical document concatenation;
- cross-document matches structurally impossible.

### R4 — Runtime Evidence

Requires:

- ordered corpus identity;
- query identity;
- deterministic runtime index commitments;
- deterministic canonical artifact;
- replay through the actual C++ runtime;
- mutation rejection.

### R5 — Self-contained Runtime Bundle

Requires:

- source documents bundled;
- runtime binaries bundled;
- replay code bundled;
- exact manifest coverage;
- payload hashes;
- bundle root;
- copied replay outside the repository;
- no external data dependency.

### R6 — Runtime Conformance Closure

R6 passes only when:

- P1-P12 pass;
- R0 is a valid historical audit;
- R1 through R5 pass;
- all runtime dependencies are present and ordered;
- no required node is missing, duplicated, skipped, or failed.

Only R6 may set:

    runtime_conformant = true
    verify_ok_permitted = true

## Top-level verifier rule

The final top-level:

    VERIFY OK

must occur only after:

    GLYPH PROOF GRAPH OK
    GLYPH RUNTIME CONFORMANCE OK

A failure in any P1-P12 or R0-R6 node must prevent `VERIFY OK`.
