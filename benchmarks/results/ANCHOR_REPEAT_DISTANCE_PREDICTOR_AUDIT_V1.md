# ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_V1

Status: measured
Date: 2026-07-01

## Purpose

Audit whether `ANCHOR_REPEAT_DISTANCE_PROFILE_V1` is a real distance signal or an in-sample / scale-count artifact.

## Decision

`anchor_distance_predictor_rejected_or_weak`

## Feature-set results

| feature set | features | in-sample best feature | feature class | in-sample accuracy | LOOCV accuracy |
|---|---:|---|---|---:|---:|
| `all_numeric` | 108 | `bytes` | `metadata` | 9/11 | 6/11 |
| `no_metadata` | 105 | `k12_all_pair_count` | `count_or_scale` | 9/11 | 4/11 |
| `normalized_or_distance_only` | 75 | `k12_useful_pair_p99` | `normalized_or_distance` | 9/11 | 3/11 |
| `no_counts_no_metadata` | 75 | `k12_useful_pair_p99` | `normalized_or_distance` | 9/11 | 3/11 |
| `count_or_scale_only` | 30 | `k12_all_pair_count` | `count_or_scale` | 9/11 | 6/11 |

## LOOCV details: normalized_or_distance_only

| held-out file | label | predicted | correct | selected feature | direction | threshold |
|---|---|---|---:|---|---|---:|
| `dickens` | bzip2 | xz | False | `k12_all_far_gt_4mb_fraction` | xz_if_ge | 0.0333503831727059 |
| `mozilla` | xz | xz | True | `k12_useful_pair_p99` | xz_if_ge | 10800992 |
| `mr` | bzip2 | xz | False | `k12_repeat_key_fraction` | xz_if_le | 0.1042571254233691 |
| `ooffice` | xz | bzip2 | False | `k12_useful_pair_p99` | xz_if_ge | 10800992 |
| `osdb` | bzip2 | xz | False | `k12_all_far_gt_4mb_fraction` | xz_if_ge | 0.0333503831727059 |
| `reymont` | bzip2 | xz | False | `k12_all_far_gt_900kb_fraction` | xz_if_ge | 0.08607435412728419 |
| `samba` | xz | bzip2 | False | `k12_all_far_gt_4mb_fraction` | xz_if_ge | 0.058333333333333334 |
| `sao` | xz | bzip2 | False | `k12_useful_pair_p99` | xz_if_ge | 10800992 |
| `webster` | xz | xz | True | `k12_useful_pair_p99` | xz_if_ge | 10800992 |
| `x-ray` | bzip2 | xz | False | `k12_repeat_key_fraction` | xz_if_le | 0.1042571254233691 |
| `xml` | bzip2 | bzip2 | True | `k12_useful_pair_p99` | xz_if_ge | 10800992 |

## Interpretation

- If LOOCV collapses, the previous 9/11 was likely in-sample threshold fitting.
- If only count/scale features work, the result is not a clean match-distance law.
- If normalized distance features survive LOOCV, then the GLYPH/STRIDE bridge remains alive.

## Non-claims

- This is still Silesia-only.
- This does not prove a production codec router.
- This does not replace out-of-sample validation.
