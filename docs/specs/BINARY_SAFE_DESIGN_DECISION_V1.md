# BINARY_SAFE_DESIGN_DECISION_V1

Status: draft  
Date: 2026-06-27

## Purpose

Record the binary-safe design decision for GLYPH exact-byte retrieval.

This decision follows:

- `docs/specs/BINARY_SAFE_ROADMAP_V1.md`
- `benchmarks/results/BINARY_SAFE_CURRENT_BOUNDARY_V1.md`

Current boundary:

    GLYPH v0.x uses a physical 0x00 sentinel.
    Source corpora containing 0x00 are currently rejected/fail.
    This is documented by BOUNDARY_PROBE_EXPECTED_FAIL.

## Decision

GLYPH binary-safe runtime will move toward a virtual sentinel model.

Data alphabet:

    0..255

Virtual boundary symbol:

    256

The virtual sentinel is not a corpus byte.

It exists only as an internal index-construction / boundary symbol.

## Why not physical 0x00

A physical 0x00 sentinel reserves one byte value and prevents indexing arbitrary binary corpora.

That prevents correct handling of:

- PE/ELF binaries
- memory dumps
- packet captures
- raw forensic images
- binary logs
- query bytes containing 0x00

Binary-safe GLYPH must treat 0x00 as normal data.

## Why not plain cyclic BWT + primary_index only

A primary_index / cyclic BWT model can remove the physical sentinel, but it risks creating boundary-crossing matches unless every query/count/locate path explicitly filters wrap-around results.

For GLYPH evidence semantics, this is dangerous.

GLYPH artifacts expose:

- exact FM interval
- exact match_count
- bounded offsets
- byte_check
- replay verifier

Therefore match_count must not include patterns that only exist because of cyclic wrap-around.

A byte_check on returned offsets is not enough if match_count itself can include boundary-crossing false positives.

## Required invariant

Binary-safe GLYPH must preserve this invariant:

    match_count == number of real non-wrapping occurrences in the source corpus

and:

    every returned offset must satisfy:
    source_corpus[offset : offset + query_len] == query_bytes

## Target binary fixture

Corpus bytes:

    41 00 42 00 43 00 42

Visualization:

    A\0B\0C\0B

Query bytes:

    00 42

Expected result:

    match_count: 2
    offsets: [1, 5]
    byte_check: true

This fixture must eventually pass through the full evidence chain:

    binary corpus with 0x00
    -> binary-safe index/runtime
    -> exact query containing 0x00
    -> FM interval/count
    -> bounded offsets
    -> byte_check
    -> artifact replay
    -> portable bundle replay
    -> schema validation
    -> ./verify.sh

## Implementation direction

The binary-safe implementation should introduce a construction path where symbols are represented as integers:

    uint16_t or uint32_t symbol stream

with:

    corpus byte b -> symbol b, where b in 0..255
    virtual sentinel -> symbol 256

The runtime/query interface remains byte-oriented:

    query bytes are still 0..255

The sentinel symbol is never accepted as query input.

## Hot-loop rule

All C++ runtime loops must be length-driven.

Forbidden assumptions:

- `strlen`
- C-string termination
- `byte == 0` means end of data
- source corpus cannot contain NUL
- query cannot contain NUL

Allowed assumptions:

- corpus has explicit byte length
- BWT/runtime has explicit symbol length
- query has explicit byte length
- sentinel is metadata/internal, not a byte

## Migration sequence

1. Keep current sentinel-safe path unchanged.
2. Add binary-safe builder path for tiny fixture.
3. Add binary-safe FM/query/count fixture.
4. Add binary-safe locate fixture.
5. Add binary-safe bounded evidence artifact.
6. Add binary-safe artifact replay.
7. Add binary-safe bundle replay.
8. Add binary-safe schema smoke validation.
9. Add binary-safe fixture to `./verify.sh`.
10. Only then claim binary-safe support.

## Non-claims

This document does not claim current GLYPH is binary-safe.

It records the selected design target for making GLYPH binary-safe.
