# SILESIA_PER_FILE_RLBWT_PROFILE_V1

Status: measured
Date: 2026-06-28

## Purpose

Measure Silesia files separately to identify which formats produce stronger BWT run signals for possible RLBWT/r-index direction.

## Boundary

Current GLYPH v0.x is sentinel-safe only. Files containing `0x00` are skipped for indexing.

## Silesia directory

`/tmp/silesia_check`

## Single-file results

| file | bytes | nul | entropy | printable | single r/n | avg run | classification | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `dickens` | 10192446 | 0 | 4.5323 | 1.0000 | 0.42920007 | 2.330 | not_run_compressible | poor |
| `mozilla` | 51220480 | 13385606 | 6.2224 | 0.3569 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `mr` | 9970564 | 5599069 | 3.6843 | 0.0901 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `nci` | 33553445 | 0 | 2.4293 | 1.0000 | 0.06544240 | 15.281 | moderately_repetitive | weak_or_moderate |
| `ooffice` | 6152192 | 766398 | 6.6399 | 0.3549 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `osdb` | 10085684 | 670202 | 6.5935 | 0.8011 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `reymont` | 6627202 | 160 | 4.8422 | 0.9961 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `samba` | 21606400 | 1098918 | 6.0945 | 0.9070 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `sao` | 7251944 | 55102 | 7.5252 | 0.4140 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `webster` | 41458703 | 0 | 4.9707 | 1.0000 | 0.29325039 | 3.410 | not_run_compressible | poor |
| `x-ray` | 8474240 | 19412 | 6.6046 | 0.2739 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |
| `xml` | 5345280 | 20970 | 5.5182 | 0.9961 | NA | NA | skipped | contains_0x00_current_glyph_v0x_boundary |

## Single-file ranking

| rank | file | single r/n | avg run | classification |
|---:|---|---:|---:|---|
| 1 | `nci` | 0.06544240 | 15.281 | moderately_repetitive |
| 2 | `webster` | 0.29325039 | 3.410 | not_run_compressible |
| 3 | `dickens` | 0.42920007 | 2.330 | not_run_compressible |

## Synthetic multiversion overlap probe

| file | version MiB | overlap shift MiB | nonoverlap r/n | overlap r/n | improvement | decision |
|---|---:|---:|---:|---:|---:|---|
| `nci` | 7 | 1 | 0.06473693 | 0.02619255 | 2.472x | multiversion_signal_alive |
| `webster` | 9 | 1 | 0.29197372 | 0.10084149 | 2.895x | no_strong_multiversion_signal |
| `dickens` | 2 | 1 | 0.43035067 | 0.27212621 | 1.581x | no_strong_multiversion_signal |

## Skipped files

| file | bytes | nul bytes | reason |
|---|---:|---:|---|
| `mozilla` | 51220480 | 13385606 | contains_0x00_current_glyph_v0x_boundary |
| `mr` | 9970564 | 5599069 | contains_0x00_current_glyph_v0x_boundary |
| `ooffice` | 6152192 | 766398 | contains_0x00_current_glyph_v0x_boundary |
| `osdb` | 10085684 | 670202 | contains_0x00_current_glyph_v0x_boundary |
| `reymont` | 6627202 | 160 | contains_0x00_current_glyph_v0x_boundary |
| `samba` | 21606400 | 1098918 | contains_0x00_current_glyph_v0x_boundary |
| `sao` | 7251944 | 55102 | contains_0x00_current_glyph_v0x_boundary |
| `x-ray` | 8474240 | 19412 | contains_0x00_current_glyph_v0x_boundary |
| `xml` | 5345280 | 20970 | contains_0x00_current_glyph_v0x_boundary |

## Interpretation

- `single r/n <= 0.05` means the file itself has strong RLBWT potential.
- `overlap r/n <= 0.05` means version-like collections for that format may be promising.
- High `r/n` means do not use that format as a compact-index argument.

## Non-claims

- This does not prove production RLBWT memory usage.
- This does not prove binary-safe production support.
- This does not prove real versioned workloads behave the same.
