# GLYPH_MATCH_DISTANCE_TAIL_PROFILE_DECISION_V1

Status: measured gate
Date: 2026-07-01

## Decision

`tail_profile_weak_borderline`

## Best rule

- feature: `global_median_distance`
- direction: `xz_if_le`
- threshold: `48`
- binary accuracy: `8/11`

## Accuracy

- bzip2 vs xz: `8/11`
- all labels including zstd19: `8/12`

## Source report

`benchmarks/results/MATCH_DISTANCE_TAIL_PROFILE_V1.md`
