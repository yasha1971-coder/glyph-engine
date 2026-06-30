# MULTIVERSION_RLBWT_PROBE_V1

Status: measured
Date: 2026-06-28

## Purpose

Test the remaining RLBWT hypothesis: RLBWT may be weak for a single corpus but useful for a collection of similar versions.

This is closer to pan-genome / many-similar-copy structure than to one isolated file.

## Important boundary

This test measures BWT run ratio only. It does not claim collection-safe retrieval semantics.

A production multiversion GLYPH would need boundary-aware document/version semantics so matches cannot falsely cross version boundaries.

## Setup

- source: `/home/glyph/GLYPH_CPP_BACKEND/benchmarks/work/named_multiversion_sources_v1/silesia_no_nul_subset.bin`
- source sha256: `bb91ea7d17fb9ef39b13f2ce154e29c612fe75b147fa0f22864e97802f86210e`
- versions: 4
- version size: 16 MiB

Two collections were built:

- `nonoverlap`: adjacent chunks, weak version similarity baseline
- `overlap`: shifted overlapping chunks, synthetic rolling-version / backup-like probe

## Results

| case | collection bytes | shift MiB | BWT bytes | runs | r/n | avg run | classification |
|---|---:|---:|---:|---:|---:|---:|---|
| nonoverlap | 67109048 | 16 | 67109049 | 13649553 | 0.20339363 | 4.917 | not_run_compressible |
| overlap | 67109036 | 2 | 67109037 | 5312749 | 0.07916593 | 12.632 | moderately_repetitive |

## Comparison

- nonoverlap r/n: `0.20339363`
- overlap r/n: `0.07916593`
- r/n improvement factor: `2.569x`

## Decision

`no_strong_multiversion_signal_at_this_scale`

Decision rule:

- If overlap collection reaches `r/n <= 0.05` and improves strongly over nonoverlap, the multiversion RLBWT hypothesis is alive.
- If not, do not spend production engineering on RLBWT.
- Even with a positive signal, a real named multi-version corpus is required before public claims.

## Non-claims

- This does not prove production RLBWT memory usage.
- This does not prove collection-safe retrieval.
- This does not prove real backup/log/version workloads will behave the same.
- This is a gate test, not a product claim.
