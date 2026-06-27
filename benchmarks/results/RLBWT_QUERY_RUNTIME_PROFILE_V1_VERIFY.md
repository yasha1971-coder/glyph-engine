# RLBWT_QUERY_RUNTIME_PROFILE_V1_VERIFY

Status: measured local runtime-profile verification  
Date: 2026-06-26

## Purpose

Validate RLBWT Query Runtime Profile V1.

This profile removes raw `bwt.bin` from the query/count path.

Runtime files:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `manifest.json`
- `rlbwt_query_runtime_manifest_v1.json`

Forbidden runtime files:

- `bwt.bin`
- `fm.bin`
- `fm_core.bin`
- `sa.bin`
- `corpus.bin`
- `corpus.sentinel.bin`
- `chunk_map.bin`

## Scope

This profile supports:

- exact query
- FM interval
- match_count

This profile does not yet support:

- locate offsets
- evidence/audit integration
- C++ runtime

Current locate status:

Existing `locate_backend_v2` still requires raw `bwt.bin` for LF steps.

Therefore this benchmark is query/count only.

## Results

| corpus | corpus_bytes | runtime_total_bytes | ratio_vs_corpus | query | FM interval | count | status |
|---|---:|---:|---:|---|---:|---:|---|
| pizza50 | 50,000,000 | 44,574,196 | 0.891x | `Ten Days that Shook the World` | [12587658, 12587659] | 1 | PASS |
| xz_cve | 38,927 | 26,909 | 0.691x | `CVE-2024-3094` | [11030, 11068] | 38 | PASS |
| synthetic_logs50 | 50,000,000 | 20,055,444 | 0.401x | `GLYPH_UNIQUE_EVENT_424242` | [20912727, 20912728] | 1 | PASS |

## File sizes

### Pizza 50MB

- `bwt.rlbwt`: 31,922,909 bytes
- `bwt.rlbwt.rank`: 12,649,620 bytes
- `manifest.json`: 736 bytes
- `rlbwt_query_runtime_manifest_v1.json`: 931 bytes
- runtime_total_bytes: 44,574,196
- ratio_vs_corpus: 0.891x

### XZ CVE corpus

- `bwt.rlbwt`: 12,790 bytes
- `bwt.rlbwt.rank`: 12,492 bytes
- `manifest.json`: 710 bytes
- `rlbwt_query_runtime_manifest_v1.json`: 917 bytes
- runtime_total_bytes: 26,909
- ratio_vs_corpus: 0.691x

### Synthetic logs 50MB

- `bwt.rlbwt`: 7,404,269 bytes
- `bwt.rlbwt.rank`: 12,649,620 bytes
- `manifest.json`: 660 bytes
- `rlbwt_query_runtime_manifest_v1.json`: 895 bytes
- runtime_total_bytes: 20,055,444
- ratio_vs_corpus: 0.401x

## Interpretation

RLBWT Query Runtime Profile V1 proves that GLYPH can perform exact query/count/FM interval retrieval from compressed BWT runtime files without raw `bwt.bin`.

This is the first GLYPH runtime profile that is below raw corpus size on all measured corpora.

Compared to Compact Runtime Profile V1:

- Compact Runtime V1 kept raw `bwt.bin`.
- RLBWT Query Runtime V1 replaces raw `bwt.bin` with `bwt.rlbwt` + `bwt.rlbwt.rank`.

Measured impact:

- Pizza 50MB moves from ~1.375x compact runtime to 0.891x query runtime.
- Synthetic logs 50MB moves from ~1.375x compact runtime to 0.401x query runtime.
- XZ CVE small corpus moves below raw corpus size despite metadata overhead.

## Boundary

This is not a full replacement for Compact Runtime Profile V1 yet.

Missing:

- locate offset recovery without raw `bwt.bin`
- C++ implementation
- query latency benchmark
- audit/evidence integration
- packaged release workflow

## Next technical target

RLBWT Query Runtime Profile V1 + locate compatibility.

Possible paths:

1. Keep `locate_core_sN.bin` and implement LF over RLBWT rank path.
2. Replace locate backend with RLBWT-aware locate.
3. Add compressed locate proof path later.

Minimum next test:

Given an FM interval from `rlbwt_fm_query_v1.py`, recover offsets using compressed LF/rank without raw BWT decode.
