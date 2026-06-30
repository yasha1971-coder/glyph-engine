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

- source: `/home/glyph/GLYPH_CPP_BACKEND/HDFS.log`
- source sha256: `0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52`
- versions: 4
- version size: 64 MiB

Two collections were built:

- `nonoverlap`: adjacent chunks, weak version similarity baseline
- `overlap`: shifted overlapping chunks, synthetic rolling-version / backup-like probe

## Results

| case | collection bytes | shift MiB | BWT bytes | runs | r/n | avg run | classification |
|---|---:|---:|---:|---:|---:|---:|---|
| nonoverlap | 268435640 | 64 | 268435641 | 28528291 | 0.10627609 | 9.409 | moderately_repetitive |
| overlap | 268435628 | 8 | 268435629 | 9301460 | 0.03465062 | 28.860 | repetitive |

## Comparison

- nonoverlap r/n: `0.10627609`
- overlap r/n: `0.03465062`
- r/n improvement factor: `3.067x`

## Decision

`multiversion_hypothesis_alive`

Decision rule:

- If overlap collection reaches `r/n <= 0.05` and improves strongly over nonoverlap, the multiversion RLBWT hypothesis is alive.
- If not, do not spend production engineering on RLBWT.
- Even with a positive signal, a real named multi-version corpus is required before public claims.

## Non-claims

- This does not prove production RLBWT memory usage.
- This does not prove collection-safe retrieval.
- This does not prove real backup/log/version workloads will behave the same.
- This is a gate test, not a product claim.
