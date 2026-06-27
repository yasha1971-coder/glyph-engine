# RLBWT_CPP_FULL_RUNTIME_VS_COMPACT_V1

Status: measured local benchmark  
Date: 2026-06-26

## Purpose

Compare current Compact Runtime Profile V1 against C++ RLBWT Full Runtime Profile V1.

This benchmark measures full query+locate:

    query bytes
    -> FM interval
    -> match_count
    -> exact offsets

## Runtime profiles compared

Compact Runtime V1:

- `bwt.bin`
- `fm_core.bin`
- `locate_core_s128.bin`

RLBWT C++ Full Runtime V1:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

## Correctness

C++ RLBWT query and locate returned expected results:

| corpus | interval | count | expected offset | status |
|---|---:|---:|---:|---|
| pizza50 | [12587658, 12587659] | 1 | 53 | PASS |
| xz_cve | [11030, 11068] | 38 | includes 274 | PASS |
| synthetic_logs50 | [20912727, 20912728] | 1 | 25000227 | PASS |

## Latency benchmark

repeats: 7

| corpus | compact_query_sec | compact_locate_sec | compact_total_sec | rlbwt_cpp_query_sec | rlbwt_cpp_locate_sec | rlbwt_cpp_total_sec | compact_total / rlbwt_total |
|---|---:|---:|---:|---:|---:|---:|---:|
| pizza50 | 0.051108 | 0.092290 | 0.143398 | 0.039707 | 0.082308 | 0.122016 | 1.175x |
| xz_cve | 0.001260 | 0.021811 | 0.023071 | 0.000949 | 0.011966 | 0.012915 | 1.786x |
| synthetic_logs50 | 0.051662 | 0.092487 | 0.144149 | 0.019968 | 0.048905 | 0.068873 | 2.093x |

## Size comparison

Previously measured deployed runtime size:

| corpus | Compact Runtime ratio | RLBWT Full Runtime ratio |
|---|---:|---:|
| pizza50 | 1.375x | 1.016x |
| xz_cve | 1.594x | 0.815x |
| synthetic_logs50 | 1.375x | 0.526x |

## Interpretation

This is the first measured GLYPH profile where the compressed runtime is both:

- smaller than Compact Runtime V1
- faster than Compact Runtime V1 on full query+locate in current CLI benchmark

The earlier Python RLBWT latency problem was implementation overhead, not a fundamental compressed-BWT limitation.

## Strategic conclusion

GLYPH has crossed the previous runtime-size bottleneck.

Current best direction:

    RLBWT Full Runtime Profile V1
    + C++ query core
    + C++ locate core

This makes GLYPH a plausible compressed exact retrieval runtime.

## Current boundary

Still missing before stronger production claim:

- persistent C++ server mode
- multi-query benchmark
- larger-corpus validation
- CI fixture
- audit/evidence integration
- refactoring duplicated RLBWT code into shared library
- rank-step/sample-step tuning for speed/size

## Next target

Create an integrated C++ RLBWT full runtime binary:

    rlbwt_full_query_locate_v1
        bwt.rlbwt
        bwt.rlbwt.rank
        locate_core_sN.bin
        query_hex
        -> interval/count/offsets

Then benchmark one-process query+locate without two separate binary invocations.
