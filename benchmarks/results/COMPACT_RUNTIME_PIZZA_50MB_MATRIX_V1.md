# COMPACT_RUNTIME_PIZZA_50MB_MATRIX_V1

Status: measured local benchmark  
Date: 2026-06-26  
Corpus: Pizza & Chili English 50MB prefix  
Corpus bytes: 50,000,000

## Purpose

Measure GLYPH Compact Runtime Profile V1 size/latency tradeoffs after separating runtime artifacts from heavy build artifacts.

This benchmark tests whether GLYPH runtime query + locate can work without:

- `fm.bin`
- `sa.bin`
- `corpus.sentinel.bin`
- `corpus.bin`
- `chunk_map.bin`

Runtime-only files:

- `bwt.bin`
- `fm_core.bin`
- `locate_core_sN.bin`
- `manifest.json`
- `compact_runtime_manifest_v1.json`

## Query

Query text:

    Ten Days that Shook the World

Expected result:

- FM interval: [12587658, 12587659]
- match_count: 1
- offset: 53

## Checkpoint-step matrix

Fixed:

- sample_step: 16

| checkpoint_step | runtime_total_bytes | ratio_vs_50mb | fm_core_bytes | locate_core_bytes | query_cli_avg_sec | locate_cli_avg_sec |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 200009836 | 4.000x | 100007960 | 50000040 | 0.214556 | 0.383224 |
| 2048 | 150009963 | 3.000x | 50008088 | 50000040 | 0.120374 | 0.300959 |
| 4096 | 125010027 | 2.500x | 25008152 | 50000040 | 0.072592 | 0.257330 |
| 8192 | 112509035 | 2.250x | 12507160 | 50000040 | 0.048937 | 0.233739 |

## Sample-step matrix

Fixed:

- checkpoint_step: 8192

| sample_step | runtime_total_bytes | ratio_vs_50mb | fm_core_bytes | locate_core_bytes | query_cli_avg_sec | locate_cli_avg_sec | locate_total_steps | locate_max_steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 112509033 | 2.250x | 12507160 | 50000040 | 0.048334 | 0.233135 | 5 | 5 |
| 32 | 87509032 | 1.750x | 12507160 | 25000040 | 0.047676 | 0.150042 | 54 | 54 |
| 64 | 75009032 | 1.500x | 12507160 | 12500040 | 0.048359 | 0.109805 | 54 | 54 |
| 128 | 68759034 | 1.375x | 12507160 | 6250040 | 0.047940 | 0.087547 | 54 | 54 |

## Best observed compact runtime candidate

Configuration:

- checkpoint_step: 8192
- sample_step: 128

Runtime footprint:

- runtime_total_bytes: 68,759,034
- ratio_vs_50mb: 1.375x
- fm_core_bytes: 12,507,160
- locate_core_bytes: 6,250,040

Correctness:

- query result: PASS
- locate result: PASS
- expected offset 53 recovered
- forbidden heavy runtime files absent

## Interpretation

This benchmark shows that the earlier 10x+ disk footprint was not a fundamental property of GLYPH runtime.

It was caused by dense FM/checkpoint layout and by keeping heavy build artifacts in the deployed directory.

Compact Runtime Profile V1 demonstrates that exact query + locate can run from compact runtime files only.

## Important caveats

This is not yet a universal industrial claim.

Limitations:

- single 50MB corpus
- single query
- CLI timings include process startup and file loading
- cold build remains heavy
- BWT remains stored uncompressed
- no r-index/PFP yet
- no production p99 latency matrix yet
- no multi-corpus validation yet

## Current conclusion

GLYPH has a credible near-term engineering path:

    dense runtime layout
    → compact runtime profile
    → size/latency matrix
    → candidate default profile
    → broader corpus validation

Measured result:

    runtime footprint reduced to 1.375x on Pizza 50MB
    while preserving exact FM interval and offset recovery.
