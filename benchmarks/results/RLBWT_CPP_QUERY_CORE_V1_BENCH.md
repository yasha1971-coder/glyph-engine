# RLBWT_CPP_QUERY_CORE_V1_BENCH

Status: measured local benchmark  
Date: 2026-06-26

## Purpose

Benchmark the first C++ RLBWT query/count core.

Tool:

    build/rlbwt_query_core_v1

Compared against Python:

    tools/rlbwt_fm_query_v1.py

## Scope

This benchmark measures query/count only:

    bwt.rlbwt
    + bwt.rlbwt.rank
    + query bytes
    -> FM interval
    -> match_count

Locate is not included in this benchmark.

## Correctness

C++ query core returned the expected FM intervals and counts:

| corpus | query | expected interval | expected count | status |
|---|---|---:|---:|---|
| pizza50 | `Ten Days that Shook the World` | [12587658, 12587659] | 1 | PASS |
| xz_cve | `CVE-2024-3094` | [11030, 11068] | 38 | PASS |
| synthetic_logs50 | `GLYPH_UNIQUE_EVENT_424242` | [20912727, 20912728] | 1 | PASS |

## Latency benchmark

repeats: 7

| corpus | python_query_avg_sec | cpp_query_avg_sec | speedup |
|---|---:|---:|---:|
| pizza50 | 4.156066 | 0.039678 | 104.75x |
| xz_cve | 0.028672 | 0.001064 | 26.95x |
| synthetic_logs50 | 1.137548 | 0.020322 | 55.98x |

## Interpretation

The Python RLBWT query path was dominated by interpreter/object overhead.

The C++ RLBWT query core reduces query latency by 26x to 105x while preserving exact FM interval correctness.

This validates the next engineering direction:

    move RLBWT runtime primitives from Python prototype to C++

## Current boundary

This is query/count only.

Still missing:

- C++ RLBWT locate
- integrated C++ query+locate benchmark
- persistent/server mode
- multi-query benchmark
- audit/evidence integration

## Next target

Implement C++ RLBWT locate core:

    FM interval
    -> LF over C++ RLBWT rank
    -> LOC1 sampled SA
    -> exact offsets

Then benchmark C++ RLBWT full query+locate against Compact Runtime V1.
