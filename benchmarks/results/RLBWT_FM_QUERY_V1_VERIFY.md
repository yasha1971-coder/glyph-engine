# RLBWT_FM_QUERY_V1_VERIFY

Status: measured local correctness test  
Date: 2026-06-26

## Purpose

Validate FM backward search over RLBWT Rank Blocks V1.

Goal:

    query bytes
    -> C array from RLBWT runs
    -> rank(c,l), rank(c,r)
    -> FM interval
    -> match_count

This verifies compressed-BWT query/count without using raw `bwt.bin` at query time.

Tool:

    tools/rlbwt_fm_query_v1.py

## Results

| corpus | query | expected_interval | expected_count | status |
|---|---|---:|---:|---|
| pizza50 | `Ten Days that Shook the World` | [12587658, 12587659] | 1 | PASS |
| xz_cve | `CVE-2024-3094` | [11030, 11068] | 38 | PASS |
| synthetic_logs50 | `GLYPH_UNIQUE_EVENT_424242` | [20912727, 20912728] | 1 | PASS |

## Meaning

RLBWT is now more than a compressed storage container.

It can support FM backward search through rank over compressed BWT.

This is the first working compressed-BWT query path in GLYPH.

## Boundary

This still does not provide full locate compatibility over compressed runtime.

Current compressed path supports:

- exact query
- FM interval
- match_count

Still missing:

- compressed locate integration
- latency benchmark
- C++ implementation
- packaged runtime profile
- audit/evidence integration

## Next step

Create RLBWT Query Runtime Profile V1:

- `rlbwt.bin`
- `rlbwt.rank`
- `locate_core_sN.bin`
- manifest

Then compare against Compact Runtime Profile V1:

- size
- query correctness
- query latency
- locate compatibility through existing locate core if interval is passed forward
