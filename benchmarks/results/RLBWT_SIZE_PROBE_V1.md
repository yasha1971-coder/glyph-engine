# RLBWT_SIZE_PROBE_V1

Status: measured local size probe  
Date: 2026-06-26

## Purpose

Measure whether raw `bwt.bin` in GLYPH Compact Runtime Profile V1 has enough run-length structure to justify a compressed BWT / RLBWT / r-index path.

This probe measures size potential only.

It does not implement rank/select over compressed BWT yet.

## Corpora

Measured corpora:

- Pizza & Chili English 50MB prefix
- XZ CVE-2024-3094 Phase 1 captured NVD corpus
- deterministic synthetic repetitive logs 50MB

## Result

| label | raw_bwt_bytes | runs | avg_run_len | max_run_len | rlbwt_u32_pair_bytes | rlbwt_varint_pair_bytes | u32_ratio_vs_bwt | varint_ratio_vs_bwt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pizza50 | 50000001 | 15954147 | 3.134 | 23984 | 79770735 | 31922853 | 1.595x | 0.638x |
| xz_cve | 38928 | 6355 | 6.126 | 325 | 31775 | 12734 | 0.816x | 0.327x |
| synthetic_logs50 | 50000001 | 3700391 | 13.512 | 490721 | 18501955 | 7404213 | 0.370x | 0.148x |

## Interpretation

Simple fixed-width `(symbol,u32 run_length)` RLBWT is not good enough for all corpora:

- Pizza 50MB becomes larger than raw BWT under u32 pairs.
- XZ and synthetic logs improve under u32 pairs.

Variable-length run encoding is promising across all measured corpora:

- Pizza 50MB: 0.638x of raw BWT
- XZ CVE corpus: 0.327x of raw BWT
- synthetic logs 50MB: 0.148x of raw BWT

## Approximate Compact Runtime impact

Current best Compact Runtime Profile V1 uses:

- checkpoint_step: 8192
- sample_step: 128

Current runtime is dominated by:

- raw `bwt.bin`
- `fm_core.bin`
- `locate_core_s128.bin`

Approximate replacement model:

    runtime ≈ rlbwt_varint_pair_bytes + fm_core_bytes + locate_core_s128_bytes

Approximate best-profile results:

| corpus | current_compact_runtime_ratio | modeled_runtime_with_varint_rlbwt |
|---|---:|---:|
| pizza50 | 1.375x | ~1.01x |
| xz_cve | 1.593x | ~0.87x |
| synthetic_logs50 | 1.375x | ~0.52x |

## Meaning

The previous runtime-bloat objection has already been reduced by Compact Runtime Profile V1.

This RLBWT probe shows a plausible path below raw-corpus size for repetitive/log-shaped corpora.

## Caveats

This is not a working compressed-BWT query engine yet.

Missing pieces:

- rank over RLBWT
- LF mapping over compressed BWT
- compatibility with locate backend
- correctness tests
- latency tests
- persistent runtime benchmark
- larger-corpus validation

## Next step

Build a minimal RLBWT container prototype:

- encode raw `bwt.bin` as `(symbol, varint_run_length)` stream
- write side metadata
- verify decode is bit-perfect
- benchmark size
- then design rank over RLBWT blocks

This should be treated as the next frontier after Compact Runtime Profile V1.
