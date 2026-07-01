# ANCHOR_REPEAT_DISTANCE_PROFILE_V1

Status: measured
Date: 2026-07-01

## Purpose

Test whether useful repeated anchors separate Silesia `bzip2` vs `xz` winners better than random-pattern match-distance.

## Method

- deterministic byte anchors sampled every `32` bytes
- k values: `[8, 12, 16, 24, 32]`
- useful anchor frequency range: `2..512`
- measures near/far repeat mass among useful anchors

## Best one-threshold rule

- feature: `k12_all_pair_count`
- direction: `xz_if_ge`
- threshold: `222876`
- binary accuracy: `9/11` = `0.8182`

## Results

| file | label | predicted | correct | bytes | entropy | NUL | k16 useful pairs | k16 far>900KB | k24 useful pairs | k24 far>900KB | k32 useful pairs | k32 far>900KB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dickens` | bzip2 | bzip2 | True | 10192446 | 4.5323 | 0 | 3240 | 0.639198 | 909 | 0.849285 | 784 | 0.931122 |
| `mozilla` | xz | xz | True | 51220480 | 6.2224 | 13385606 | 244167 | 0.282323 | 181115 | 0.278613 | 132188 | 0.314000 |
| `mr` | bzip2 | bzip2 | True | 9970564 | 3.6843 | 5599069 | 7876 | 0.246064 | 1723 | 0.448636 | 734 | 0.286104 |
| `nci` | zstd19 | out_of_binary_gate | False | 33553445 | 2.4293 | 0 | 360171 | 0.356689 | 401639 | 0.454483 | 392914 | 0.489692 |
| `ooffice` | xz | bzip2 | False | 6152192 | 6.6399 | 766398 | 10717 | 0.094336 | 4937 | 0.092971 | 3767 | 0.095301 |
| `osdb` | bzip2 | bzip2 | True | 10085684 | 6.5935 | 670202 | 87673 | 0.272535 | 35798 | 0.350746 | 21212 | 0.306572 |
| `reymont` | bzip2 | bzip2 | True | 6627202 | 4.8422 | 160 | 32167 | 0.314297 | 11364 | 0.316790 | 4014 | 0.285999 |
| `samba` | xz | xz | True | 21606400 | 6.0945 | 1098918 | 109021 | 0.116445 | 75071 | 0.108191 | 53729 | 0.116771 |
| `sao` | xz | bzip2 | False | 7251944 | 7.5252 | 55102 | 0 | 0.000000 | 0 | 0.000000 | 0 | 0.000000 |
| `webster` | xz | xz | True | 41458703 | 4.9707 | 0 | 259920 | 0.386654 | 184377 | 0.426604 | 114050 | 0.484814 |
| `x-ray` | bzip2 | bzip2 | True | 8474240 | 6.6046 | 19412 | 2 | 0.000000 | 0 | 0.000000 | 0 | 0.000000 |
| `xml` | bzip2 | bzip2 | True | 5345280 | 5.5182 | 20970 | 88999 | 0.004225 | 72227 | 0.006853 | 60563 | 0.009181 |

## Decision

`anchor_repeat_distance_borderline_alive`

- bzip2 vs xz accuracy: `9/11`
- all labels including zstd19 outlier: `9/12`

## Interpretation

- If this beats the random-pattern tail profile, the useful-anchor hypothesis survives.
- If it does not, match-distance is not yet a reliable codec predictor.
- A strong result still needs out-of-sample validation, because this is Silesia in-sample threshold search.

## Non-claims

- This is not a production codec router.
- This is not binary-safe GLYPH production support.
- This does not replace compressor trials.
- This is a falsification gate.
