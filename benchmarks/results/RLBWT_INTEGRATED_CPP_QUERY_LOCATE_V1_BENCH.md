# RLBWT_INTEGRATED_CPP_QUERY_LOCATE_V1_BENCH

Status: measured local benchmark  
Date: 2026-06-26

## Purpose

Benchmark integrated C++ RLBWT query+locate runtime against two separate C++ binary invocations.

Separate path:

    rlbwt_query_core_v1
    +
    rlbwt_locate_core_v1

Integrated path:

    rlbwt_full_query_locate_v1

The integrated binary loads:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

and directly returns:

- FM interval
- match_count
- exact offsets

## Correctness

Integrated binary returned expected exact results:

| corpus | interval | count | expected offset | status |
|---|---:|---:|---:|---|
| pizza50 | [12587658, 12587659] | 1 | 53 | PASS |
| xz_cve | [11030, 11068] | 38 | includes 274 | PASS |
| synthetic_logs50 | [20912727, 20912728] | 1 | 25000227 | PASS |

## Latency benchmark

repeats: 7

| corpus | separate_query_sec | separate_locate_sec | separate_total_sec | integrated_total_sec | separate / integrated |
|---|---:|---:|---:|---:|---:|
| pizza50 | 0.039474 | 0.081689 | 0.121164 | 0.068556 | 1.767x |
| xz_cve | 0.001094 | 0.011710 | 0.012804 | 0.008274 | 1.547x |
| synthetic_logs50 | 0.019758 | 0.048003 | 0.067761 | 0.044903 | 1.509x |

## Interpretation

The integrated C++ RLBWT full runtime removes duplicated process startup and duplicated structure loading.

It is faster than separate C++ query + locate binaries on all measured corpora.

This confirms the next architecture direction:

    one loaded runtime
    one query path
    shared RLBWT/rank/locate structures

## Strategic meaning

GLYPH now has a compressed-BWT runtime path that is:

- packaged below or near raw corpus size
- exact for query/count/locate
- implemented in C++
- faster when integrated into one process

## Current boundary

Still missing:

- persistent server mode
- multi-query benchmark
- larger-corpus validation
- CI fixture
- evidence/audit integration
- refactoring duplicated RLBWT code into shared library

## Next target

Create persistent C++ RLBWT runtime server:

    rlbwt_full_query_locate_server_v1

Goal:

- load runtime files once
- accept multiple query_hex requests
- return interval/count/offsets
- benchmark warm query latency
