# FASTQ_RLBWT_VIABILITY_PROBE_V1

Status: measured
Date: 2026-06-28

## Purpose

Measure whether real FASTQ corpora have low BWT run ratio `r/n`, which would justify further RLBWT/r-index engineering for GLYPH.

This is a viability gate, not a production RLBWT claim.

## Results

| source | sample | sample bytes | raw entropy | nul bytes | dup64 | dup256 | dup1024 | BWT r/n | avg run | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `NA12878_1gb.fastq` | 256 MiB | 268435456 | 2.9285 | 0 | 0.305218 | 0.000000 | 0.000000 | 0.15165326 | 6.594 | poor RLBWT signal |

## Decision rule

- `r/n <= 0.01`: strong RLBWT direction.
- `r/n <= 0.05`: worth continuing.
- `r/n > 0.15`: do not use this corpus as a compact-index argument.

## Non-claims

- This does not build a production RLBWT index.
- This does not claim FASTQ support at 50 GB full scale.
- This does not claim binary-safe production runtime.
- This only measures real-corpus repetitiveness signals for future GLYPH engineering.

## Next step

If 256 MiB shows `r/n <= 0.05`, rerun the same corpus with 1024 MiB before committing to RLBWT engineering.
