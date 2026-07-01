# GLYPH_ANCHOR_REPEAT_DISTANCE_PROFILE_DECISION_V1

Status: measured gate
Date: 2026-07-01

## Decision

`anchor_repeat_distance_borderline_alive`

## Best rule

- feature: `k12_all_pair_count`
- direction: `xz_if_ge`
- threshold: `222876`
- binary accuracy: `9/11`

## Accuracy

- bzip2 vs xz: `9/11`
- all labels including zstd19: `9/12`

## Source report

`benchmarks/results/ANCHOR_REPEAT_DISTANCE_PROFILE_V1.md`
