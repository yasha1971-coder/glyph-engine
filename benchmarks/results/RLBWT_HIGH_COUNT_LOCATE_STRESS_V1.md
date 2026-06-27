# RLBWT_HIGH_COUNT_LOCATE_STRESS_V1

Status: measured local benchmark  
Date: 2026-06-27

## Purpose

Measure high-count locate stress for C++ RLBWT full query+locate runtime.

Tool:

    rlbwt_full_query_locate_summary_v1

Runtime files:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

This benchmark suppresses printing all offsets.

It measures:

- FM interval
- match_count
- total LF steps
- max LF steps
- wall time

## Why summary mode exists

The normal integrated runtime prints every recovered offset.

For high-count queries this mixes three costs:

1. FM query
2. LF locate work
3. huge stdout offset payload

Summary mode suppresses offset arrays and prints only first/last offset.

## Results

| corpus | query | match_count | total_steps | max_steps | first_offset | last_offset | elapsed_sec |
|---|---|---:|---:|---:|---:|---:|---:|
| pizza50 | `EVENT` | 24 | 2,090 | 256 | 651,167 | 38,001,514 | 0.07 |
| pizza50 | `INFO` | 27 | 3,917 | 553 | 787,652 | 28,565,852 | 0.08 |
| synthetic_logs50 | `to` | 98,144 | 12,708,871 | 1,493 | 223 | 49,999,643 | 10.87 |
| synthetic_logs50 | `INFO` | 294,433 | 37,846,029 | 1,510 | 39 | 49,999,966 | 33.81 |
| pizza50 | `of` | 312,581 | 39,692,212 | 1,609 | 206 | 49,999,752 | 131.04 |
| pizza50 | `the` | 761,402 | 96,600,139 | 1,699 | 73 | 49,999,994 | 322.21 |

## Interpretation

C++ RLBWT query/count is no longer the main bottleneck.

Persistent server removes process/load overhead for warm workloads.

The hard remaining bottleneck is high-count locate.

Latency scales with:

- match_count
- total LF steps
- sampled locate density
- RLBWT rank/symbol lookup cost
- corpus/run structure

Small multi-hit queries are acceptable:

- pizza `EVENT`: 24 hits, 2,090 LF steps, 0.07 sec
- pizza `INFO`: 27 hits, 3,917 LF steps, 0.08 sec

Large high-count queries become expensive:

- synthetic `INFO`: 294,433 hits, 37.8M LF steps, 33.81 sec
- pizza `the`: 761,402 hits, 96.6M LF steps, 322.21 sec

## Strategic conclusion

GLYPH compressed runtime has crossed the disk/runtime query bottleneck.

The next bottleneck is not exact search.

The next bottleneck is exhaustive offset recovery for very high-count queries.

For audit/evidence workloads this may be acceptable when evidence only needs:

- match_count
- FM interval
- limited offset sample
- first N offsets
- bounded locate
- paginated locate

For exhaustive grep-like workloads, high-count locate requires optimization.

## Next technical targets

1. Add bounded locate mode:

       --max-offsets N

2. Add count-only mode:

       query -> FM interval + match_count, no locate

3. Benchmark sample_step tradeoff:

       sample_step 16 / 32 / 64 / 128

4. Add paginated locate:

       interval + start + limit

5. Investigate faster LF/rank implementation and shared library refactor.
