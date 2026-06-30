# NAMED_MULTIVERSION_RLBWT_CORPORA_V1

Status: measured
Date: 2026-06-28

## Purpose

Run the multiversion RLBWT gate on named control corpora available on OVH.

This tests whether overlapping version-like collections reduce BWT run ratio compared to non-overlapping chunks.

## Results

| corpus | version MiB | overlap shift MiB | nonoverlap r/n | overlap r/n | improvement | decision |
|---|---:|---:|---:|---:|---:|---|
| enwik9 | 64 | 8 | 0.34833348 | 0.12658196 | 2.752x | no_strong_multiversion_signal_at_this_scale |
| silesia_no_nul_subset | 16 | 2 | 0.20339363 | 0.07916593 | 2.569x | no_strong_multiversion_signal_at_this_scale |

## Interpretation

- If overlap `r/n <= 0.05`, the multiversion RLBWT hypothesis remains alive for that corpus family.
- If overlap remains high, RLBWT should not be prioritized for that corpus.
- These are synthetic overlap/version probes, not real backup/document-version corpora.

## Non-claims

- This does not prove production RLBWT memory usage.
- This does not prove collection-safe retrieval semantics.
- This does not replace a real named versioned corpus test.
