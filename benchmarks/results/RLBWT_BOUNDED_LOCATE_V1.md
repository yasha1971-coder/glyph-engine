# RLBWT_BOUNDED_LOCATE_V1

Status: measured local benchmark  
Date: 2026-06-27

## Purpose

Validate bounded locate mode for C++ RLBWT full query+locate runtime.

Tool:

    rlbwt_full_query_locate_limit_v1

Runtime files:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

Bounded locate mode returns:

- FM interval
- full match_count
- total_possible_count
- only first N located offsets
- bounded=true when N < match_count

This is important for evidence/audit workloads where full exhaustive offset recovery is often unnecessary.

## Results

| corpus | query | max_offsets | match_count | located_count | bounded | total_steps_returned | max_steps_returned | elapsed_sec |
|---|---|---:|---:|---:|---|---:|---:|---:|
| pizza50 | `Ten Days that Shook the World` | 10 | 1 | 1 | false | 54 | 54 | 0.07 |
| pizza50 | absent query | 10 | 0 | 0 | false | 0 | 0 | 0.07 |
| pizza50 | `the` | 0 | 761402 | 0 | true | 0 | 0 | 0.06 |
| pizza50 | `the` | 10 | 761402 | 10 | true | 1596 | 429 | 0.07 |
| pizza50 | `the` | 100 | 761402 | 100 | true | 13433 | 596 | 0.11 |
| synthetic_logs50 | `INFO` | 0 | 294433 | 0 | true | 0 | 0 | 0.04 |
| synthetic_logs50 | `INFO` | 10 | 294433 | 10 | true | 1759 | 502 | 0.04 |
| synthetic_logs50 | `INFO` | 100 | 294433 | 100 | true | 13013 | 617 | 0.05 |

## Comparison with exhaustive locate

Previously measured exhaustive high-count locate:

| corpus | query | exhaustive_count | exhaustive_steps | exhaustive_elapsed_sec |
|---|---|---:|---:|---:|
| pizza50 | `the` | 761402 | 96600139 | 322.21 |
| synthetic_logs50 | `INFO` | 294433 | 37846029 | 33.81 |

Bounded locate changes high-count behavior:

- `pizza50/the`:
  - exhaustive: 322.21 sec
  - max_offsets=100: 0.11 sec

- `synthetic_logs50/INFO`:
  - exhaustive: 33.81 sec
  - max_offsets=100: 0.05 sec

## Interpretation

Bounded locate separates search correctness from exhaustive offset enumeration.

The runtime can now answer:

    "How many exact matches exist?"

without paying for all offsets.

It can also return a bounded evidence sample:

    first N offsets

while preserving the full FM interval and full match_count.

## Strategic meaning

This is the practical evidence-runtime mode GLYPH needed.

For high-count queries, exhaustive locate is a grep-like workload.

For audit/evidence workloads, bounded locate is often enough:

- full count
- deterministic interval
- limited reproducible offsets
- bounded latency

## Current boundary

The current bounded mode locates the first N positions in FM interval order, then sorts offsets.

Still missing:

- paginated locate
- offset range windows
- server support for max_offsets
- audit/evidence integration
- stable CLI spec
- CI fixture

## Next target

Add bounded locate support to persistent server protocol:

    query_hex<TAB>max_offsets

so warm server can return count + bounded offsets without exhaustive locate.
