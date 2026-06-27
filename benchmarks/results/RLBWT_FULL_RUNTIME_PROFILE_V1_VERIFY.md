# RLBWT_FULL_RUNTIME_PROFILE_V1_VERIFY

Status: measured local full-runtime verification  
Date: 2026-06-26

## Purpose

Validate RLBWT Full Runtime Profile V1.

This profile packages compressed query/count/locate runtime into one deployed runtime directory.

Runtime files:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`
- `manifest.json`
- `rlbwt_full_runtime_manifest_v1.json`

Forbidden runtime files:

- `bwt.bin`
- `fm.bin`
- `fm_core.bin`
- `sa.bin`
- `corpus.bin`
- `corpus.sentinel.bin`
- `chunk_map.bin`

## Supported

- exact query
- FM interval
- match_count
- exact offset recovery

## Results

| corpus | corpus_bytes | runtime_total_bytes | ratio_vs_corpus | query | FM interval | count | offsets | status |
|---|---:|---:|---:|---|---:|---:|---|---|
| pizza50 | 50,000,000 | 50,824,139 | 1.016x | `Ten Days that Shook the World` | [12587658, 12587659] | 1 | [53] | PASS |
| xz_cve | 38,927 | 31,713 | 0.815x | `CVE-2024-3094` | [11030, 11068] | 38 | 38 offsets, includes 274 | PASS |
| synthetic_logs50 | 50,000,000 | 26,305,387 | 0.526x | `GLYPH_UNIQUE_EVENT_424242` | [20912727, 20912728] | 1 | [25000227] | PASS |

## File sizes

### Pizza 50MB

- `bwt.rlbwt`: 31,922,909
- `bwt.rlbwt.rank`: 12,649,620
- `locate_core_s128.bin`: 6,250,040
- `manifest.json`: 736
- `rlbwt_full_runtime_manifest_v1.json`: 834
- runtime_total_bytes: 50,824,139
- ratio_vs_corpus: 1.016x

### XZ CVE corpus

- `bwt.rlbwt`: 12,790
- `bwt.rlbwt.rank`: 12,492
- `locate_core_s128.bin`: 4,904
- `manifest.json`: 710
- `rlbwt_full_runtime_manifest_v1.json`: 817
- runtime_total_bytes: 31,713
- ratio_vs_corpus: 0.815x

### Synthetic logs 50MB

- `bwt.rlbwt`: 7,404,269
- `bwt.rlbwt.rank`: 12,649,620
- `locate_core_s128.bin`: 6,250,040
- `manifest.json`: 660
- `rlbwt_full_runtime_manifest_v1.json`: 798
- runtime_total_bytes: 26,305,387
- ratio_vs_corpus: 0.526x

## Interpretation

RLBWT Full Runtime Profile V1 is the first GLYPH profile that packages compressed query/count/locate correctness without raw BWT runtime.

This moves GLYPH from:

    Compact Runtime V1 with raw bwt.bin

to:

    compressed BWT full runtime prototype

The old runtime bloat objection is no longer structurally true.

Measured result:

- Pizza 50MB is near raw corpus size: 1.016x
- Synthetic logs 50MB is far below raw corpus size: 0.526x
- XZ CVE small corpus is below raw corpus size despite metadata overhead: 0.815x

## Boundary

This is still a Python prototype.

Missing before stronger production claim:

- C++ implementation
- latency benchmark
- multi-query validation
- larger-corpus validation
- audit/evidence integration
- CI test fixture
- compact profile documentation

## Next technical target

Benchmark RLBWT Full Runtime Profile V1 latency and compare against Compact Runtime Profile V1.

Then implement a C++ RLBWT runtime core.
