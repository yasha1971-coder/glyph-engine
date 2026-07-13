# GLYPH_RUNTIME_CONFORMANCE_V1

Status: normative implementation target  
Version: 1  
Date: 2026-07-13

## Purpose

Define conformance between the executable GLYPH C++ runtime and the verified
reference semantics established by GLYPH_PROOF_GRAPH_V1.

The proof graph defines what correctness means.

Runtime conformance proves that the actual compiled implementation preserves
those semantics.

## Current baseline

The current sentinel-safe v0.x runtime:

- accepts byte values 0x01 through 0xFF;
- rejects source corpora containing 0x00;
- appends a physical 0x00 sentinel;
- stores BWT symbols as uint8 values;
- stores C and rank state for 256 symbols;
- accepts byte-oriented hexadecimal queries.

This profile is not fully conformant to the P1-P12 virtual-sentinel model.

## Required target model

The conformant runtime must represent:

    source byte symbols: 0..255
    virtual sentinel:    256

The virtual sentinel must not be encoded as a source byte.

## Runtime obligations

### R1 — Input domain

Every byte value from 0x00 through 0xFF must be accepted as source data.

### R2 — Sentinel separation

The virtual sentinel must be represented independently from all source bytes.

A query for byte 0x00 must search source byte 0x00 only.

The sentinel must not be queryable through the byte-query API.

### R3 — Suffix-array cardinality

For a single document containing n bytes, the runtime suffix array must contain:

    n + 1 rows

including the terminal suffix.

### R4 — BWT symbol domain

The runtime BWT must represent symbols 0 through 256 without collision.

A byte-oriented BWT file containing only uint8 values is insufficient unless
the sentinel location is stored separately and normatively bound.

### R5 — FM alphabet

C, histogram, rank, checkpoints, and LF must operate over 257 symbols or an
equivalent collision-free representation.

### R6 — Binary-safe query transport

Queries remain arbitrary non-empty byte strings encoded as canonical lowercase
hexadecimal.

All bytes 0x00 through 0xFF must survive transport exactly.

### R7 — Exact count

For every fixture, C++ FM count must equal the independent byte oracle.

### R8 — Exact locate

Returned coordinates must equal canonical document-local coordinates and must
pass byte verification.

### R9 — Boundary semantics

No query may match across document boundaries.

### R10 — Replay and bundle identity

Artifacts and bundles must record the runtime format version and remain
deterministically replayable.

### R11 — Mutation rejection

The runtime gate must reject:

- sentinel/0x00 collision;
- omitted terminal suffix;
- malformed query hex;
- incorrect count;
- incorrect coordinate;
- cross-document match;
- runtime-format mismatch.

### R12 — Proof-graph integration

Top-level VERIFY OK may claim binary-safe runtime conformance only after the
compiled C++ runtime passes this specification.

## Mandatory fixtures

- ASCII text;
- embedded 0x00;
- byte 0xFF;
- invalid UTF-8;
- repeated 0x00;
- repeated matches;
- zero matches;
- all 256 byte values;
- query beginning with 0x00;
- query ending with 0x00;
- query containing 0x00 internally;
- document-boundary control.

## Baseline semantics

A baseline audit may report:

    audit_ok = true
    runtime_conformant = false

This means the audit executed correctly and identified implementation gaps.

It must not be presented as runtime conformance.

## Completion criterion

GLYPH_RUNTIME_CONFORMANCE_V1 is complete only when the actual compiled C++
build/query/locate/replay path passes all mandatory fixtures and is included
before the final VERIFY OK.
