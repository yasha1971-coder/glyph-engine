# RLBWT_RANK_BLOCKS_V1_VERIFY

Status: measured local correctness test  
Date: 2026-06-26

## Purpose

Validate RLBWT Rank Blocks V1.

Goal:

    rank(c, pos) over RLBWT without fully decoding bwt.bin

This is the first step from compressed BWT container toward compressed FM-index runtime.

## Method

For each corpus:

1. Build RLBWT container from raw `bwt.bin`.
2. Build RLBWT rank block index with `rank_step=8192`.
3. Verify `rank(c,pos)` against raw BWT counts.
4. Run 1000 random checks plus fixed edge-position checks.
5. Require zero mismatches.

Tool:

    tools/rlbwt_rank_blocks_v1.py

## Results

| corpus | raw_bwt_bytes | rlbwt_bytes | rank_index_bytes | combined_bytes | combined_ratio_vs_bwt | checks | bad |
|---|---:|---:|---:|---:|---:|---:|---:|
| pizza50 | 50,000,001 | 31,922,909 | 12,649,620 | 44,572,529 | 0.891x | 1060 | 0 |
| xz_cve | 38,928 | 12,790 | 12,492 | 25,282 | 0.649x | 1060 | 0 |
| synthetic_logs50 | 50,000,001 | 7,404,269 | 12,649,620 | 20,053,889 | 0.401x | 1060 | 0 |

## Interpretation

RLBWT is no longer only a compressed storage artifact.

RLBWT Rank Blocks V1 can answer rank queries over the compressed BWT representation without full BWT decode.

This makes compressed-BWT backward search technically reachable.

## Important result

Even after adding rank blocks:

- Pizza50 remains below raw BWT size: 0.891x
- XZ CVE remains below raw BWT size: 0.649x
- Synthetic logs50 becomes much smaller than raw BWT: 0.401x

This confirms the RLBWT path is physically meaningful.

## Boundary

This is not yet a complete compressed FM-index.

Still missing:

- compressed backward search using RLBWT rank
- C array integration
- FM interval correctness tests
- locate compatibility
- latency benchmarks
- C++ implementation

## Next step

Implement compressed-BWT backward search prototype:

    query bytes
    → C array
    → RLBWT rank(c,l), rank(c,r)
    → FM interval
    → match_count

Target validation:

The compressed-BWT FM interval must match existing raw-BWT/FM results for:

- Pizza query: `Ten Days that Shook the World`
- XZ query: `CVE-2024-3094`
- synthetic log query: `GLYPH_UNIQUE_EVENT_424242`
