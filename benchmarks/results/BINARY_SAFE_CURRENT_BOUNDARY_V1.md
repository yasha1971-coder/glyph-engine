# BINARY_SAFE_CURRENT_BOUNDARY_V1

Status: expected boundary failure  
Date: 2026-06-27

## Purpose

Document the current GLYPH v0.x binary-safety boundary.

Current invariant:

    source corpus must not contain 0x00

Boundary probe:

    tools/run_binary_safe_boundary_probe_v1.sh

## Probe corpus

Bytes:

    41 00 42 00 43 00 42

Visualization:

    A\0B\0C\0B

Future binary-safe query:

    00 42

Future expected result:

- match_count: 2
- offsets: [1, 5]
- byte_check: true

## Current result

The current sentinel-safe index builder does not support source corpus containing 0x00.

The boundary probe is expected to fail at current v0.x indexing/preparation stage.

Expected probe result:

    BOUNDARY_PROBE_EXPECTED_FAIL

## Interpretation

This is not a regression.

This is a documented boundary before implementing binary-safe GLYPH.

## Required future fix

Binary-safe GLYPH must replace the physical 0x00 sentinel model with:

- virtual sentinel outside byte alphabet, or
- coordinate boundary / primary_index model with strict no-wrap retrieval semantics

The final binary-safe fixture must pass the full chain:

    binary corpus with 0x00
    -> exact query containing 0x00
    -> exact match_count
    -> bounded offsets
    -> byte_check
    -> artifact replay
    -> bundle replay
    -> schema validation
    -> ./verify.sh
