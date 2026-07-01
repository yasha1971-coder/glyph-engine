# MATCH_DISTANCE_PROFILE_V1

Status: measured
Date: 2026-07-01

## Purpose

Test whether GLYPH-derived match-distance separates `reymont` and `webster`, where entropy/r/n/BWT-r/n/profile-std failed.

## Boundary

Current GLYPH v0.x is sentinel-safe. Files with `0x00` are stripped in working copies and recorded.

## Results

| file | expected | predicted | correct | source bytes | NUL | prepared bytes | patterns | median distance | p90 median | far>900KB median | near<=64KB median |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reymont` | bzip2 | bzip2 | True | 6627202 | 160 | 6627042 | 300 | 123610.000000 | 1874387.900000 | 0.000000 | 0.323055 |
| `webster` | xz | bzip2 | False | 41458703 | 0 | 41458703 | 300 | 361163.750000 | 15335004.900000 | 0.280154 | 0.244048 |

## Per-k summary

### reymont

| k | patterns | median distance | far>900KB median | prediction |
|---:|---:|---:|---:|---|
| 8 | 100 | 48435.25 | 0.0 | bzip2 |
| 12 | 100 | 125180.25 | 0.02181050656660413 | bzip2 |
| 16 | 100 | 446482.0 | 0.13392857142857142 | bzip2 |

### webster

| k | patterns | median distance | far>900KB median | prediction |
|---:|---:|---:|---:|---|
| 8 | 100 | 203183.75 | 0.2297551789077213 | bzip2 |
| 12 | 100 | 655799.5 | 0.4193693693693694 | bzip2 |
| 16 | 100 | 408195.0 | 0.2361111111111111 | bzip2 |

## Interpretation

Known-label gate accuracy: `1/2`.

If this separates `reymont` and `webster`, match-distance remains alive as a structural feature.
If it does not, do not build claims around it.

## Non-claims

- This is not a production codec router.
- This is not binary-safe GLYPH production support.
- This does not replace compressor trials.
- This is a falsification gate.
