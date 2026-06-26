# COMPACT_RUNTIME_PROFILE_V1

Status: experimental  
Date: 2026-06-26

## Purpose

Define the first compact runtime profile for GLYPH.

This profile separates heavy build artifacts from deployed runtime artifacts.

The goal is to reduce runtime disk footprint while preserving:

- exact-byte query
- FM interval
- match count
- sampled locate offsets
- byte-check/evidence compatibility

## Problem

The earlier public evidence path kept dense and temporary build artifacts in the same directory:

- `fm.bin`
- `sa.bin`
- `corpus.sentinel.bin`
- `corpus.bin`
- `chunk_map.bin`

This made GLYPH appear to require 8x-12x disk overhead.

That is not a fundamental requirement of query runtime.

It is a layout/profile problem.

## Runtime artifact set

Compact Runtime Profile V1 keeps only:

- `bwt.bin`
- `fm_core.bin`
- `locate_core_s16.bin`
- `manifest.json`
- optional `compact_runtime_manifest_v1.json`

## Forbidden runtime artifacts

The following files are build artifacts and must not be required in a compact runtime directory:

- `fm.bin`
- `sa.bin`
- `corpus.sentinel.bin`
- `corpus.bin`
- `chunk_map.bin`

## Query path

Query/count/FM interval runs from:

- `fm_core.bin`
- `bwt.bin`

using:

- `query_fm_core_v1`

Dense `fm.bin` is not required.

## Locate path

Offset recovery runs from:

- `fm_core.bin`
- `locate_core_s16.bin`
- `bwt.bin`

using:

- `locate_backend_v2`
- `glyph_locate_offsets_v0.py`

Full `sa.bin` is not required at runtime.

## Local validation: Pizza 50MB

Source demo directory:

    examples/public-evidence-demo/work/pizza_english_50mb

Compact runtime candidate:

    /tmp/glyph_compact_runtime_pizza_50mb_c2048_s16_runtime_only

Build parameters:

- checkpoint_step: 2048
- sample_step: 16

Runtime files:

- `bwt.bin`: 50,000,001 bytes
- `fm_core.bin`: 50,008,088 bytes
- `locate_core_s16.bin`: 50,000,040 bytes
- `manifest.json`: 736 bytes

Runtime total:

- 150,008,865 bytes

Ratio vs original 50MB corpus:

- about 3.0x

Forbidden files were absent:

- `fm.bin`
- `sa.bin`
- `corpus.sentinel.bin`
- `corpus.bin`
- `chunk_map.bin`

Query:

    Ten Days that Shook the World

Observed compact query result:

- bwt_bytes: 50000001
- checkpoint_step: 2048
- num_checkpoints: 24416
- fm_interval: [12587658, 12587659]
- match_count: 1
- count: 1

Observed compact locate result:

- offset_mode: locate_backend_v2
- offsets: [53]
- total_steps: 5
- max_steps: 5
- ok: true

## Meaning

This proves that dense `fm.bin` is not required for query runtime.

It also proves that full `sa.bin` is not required for locate runtime.

The next engineering objective is to measure the size/latency tradeoff over:

- checkpoint_step: 1024, 2048, 4096, 8192
- sample_step: 16, 32, 64, 128

## Non-claims

This profile does not yet solve:

- cold build time
- full compressed BWT storage
- r-index
- PFP construction
- cryptographic inclusion proof
- non-membership proof
- verifier without corpus or commitment

## Current conclusion

GLYPH's 10x+ disk bloat was not a fundamental law.

It was caused by dense runtime layout.

Compact Runtime Profile V1 is the first practical path toward a smaller industrial runtime profile.
