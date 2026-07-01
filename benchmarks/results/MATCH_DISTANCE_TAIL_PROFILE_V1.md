# MATCH_DISTANCE_TAIL_PROFILE_V1

Status: measured
Date: 2026-07-01

## Purpose

Test whether match-distance tail shape separates Silesia `bzip2` vs `xz` winners after median match-distance failed.

## Boundary

This is a raw-byte feature gate over full Silesia files. It is not a GLYPH v0.x production binary-safe claim.

Current GLYPH v0.x remains sentinel-safe; this test measures whether the feature is worth integrating into a future binary-safe GLYPH/STRIDE path.

## Best one-threshold rule

- feature: `global_median_distance`
- direction: `xz_if_le`
- threshold: `48`
- binary accuracy: `8/11` = `0.7273`

## Results

| file | label | predicted | correct | bytes | entropy | NUL | patterns | median | p90 | p99 | far>900KB diff | far>4MB diff | near<=64KB diff | p90/median |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dickens` | bzip2 | bzip2 | True | 10192446 | 4.5323 | 0 | 240 | 16213 | 302480 | 2724945 | 0.040194 | 0.004841 | 0.736546 | 18.656633565657188 |
| `mozilla` | xz | xz | True | 51220480 | 6.2224 | 13385606 | 240 | 48 | 4008 | 248092 | 0.002996 | 0.000599 | 0.979176 | 83.5 |
| `mr` | bzip2 | xz | False | 9970564 | 3.6843 | 5599069 | 240 | 1 | 1024 | 505472 | 0.003845 | 0.000560 | 0.973139 | 1024.0 |
| `nci` | zstd19 | out_of_binary_gate | False | 33553445 | 2.4293 | 0 | 240 | 70 | 3640 | 33882 | 0.000250 | 0.000032 | 0.994611 | 52.0 |
| `ooffice` | xz | xz | True | 6152192 | 6.6399 | 766398 | 240 | 1 | 1672 | 57712 | 0.000747 | 0.000021 | 0.990905 | 1672.0 |
| `osdb` | bzip2 | bzip2 | True | 10085684 | 6.5935 | 670202 | 240 | 6631 | 25323 | 149385 | 0.002126 | 0.000070 | 0.980336 | 3.8188810134218065 |
| `reymont` | bzip2 | bzip2 | True | 6627202 | 4.8422 | 160 | 240 | 478 | 6087 | 65292 | 0.000797 | 0.000051 | 0.989643 | 12.734309623430962 |
| `samba` | xz | xz | True | 21606400 | 6.0945 | 1098918 | 240 | 38 | 49 | 12979 | 0.000409 | 0.000092 | 0.997514 | 1.2894736842105263 |
| `sao` | xz | bzip2 | False | 7251944 | 7.5252 | 55102 | 160 | 37632 | 332192 | 2320276 | 0.038227 | 0.002368 | 0.633965 | 8.827380952380953 |
| `webster` | xz | bzip2 | False | 41458703 | 4.9707 | 0 | 240 | 459 | 5007 | 59926 | 0.000834 | 0.000150 | 0.990854 | 10.908496732026144 |
| `x-ray` | bzip2 | bzip2 | True | 8474240 | 6.6046 | 19412 | 80 | 1334080 | 6574294 | 7878620 | 0.595506 | 0.213483 | 0.033708 | 4.92796084192852 |
| `xml` | bzip2 | bzip2 | True | 5345280 | 5.5182 | 20970 | 240 | 134 | 1415 | 9807 | 0.000020 | 0.000000 | 0.998643 | 10.559701492537313 |

## Decision

`tail_profile_weak_borderline`

- binary gate accuracy: `8/11`
- all-label accuracy including `zstd19` outlier: `8/12`

## Interpretation

- Median match-distance alone already failed.
- This test checks whether the tail of match-distance distribution is a better structural feature.
- If accuracy is high, the GLYPH/STRIDE bridge remains alive as tail-distance profiling.
- If accuracy is weak, do not claim codec prediction from match-distance yet.

## Non-claims

- This is not a production codec router.
- This is not binary-safe GLYPH production support.
- This does not replace compressor trials.
- This is an in-sample falsification gate over Silesia.
