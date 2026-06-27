# RLBWT_BOUNDED_SERVER_V1

Status: measured local benchmark  
Date: 2026-06-27

## Purpose

Validate bounded locate support in persistent C++ RLBWT server.

Server protocol:

    query_hex<TAB>max_offsets

Server output:

    OK<TAB>l<TAB>r<TAB>match_count<TAB>located_count<TAB>bounded<TAB>total_steps<TAB>max_steps<TAB>offsets

This allows GLYPH to return full exact count and bounded deterministic offset evidence without exhaustive locate.

| corpus | query | max_offsets | match_count | located_count | bounded | avg_sec | p50_sec | p95_sec | p99_sec | min_sec | max_sec | total_steps | max_steps |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pizza50 | `the` | 0 | 761402 | 0 | true | 0.000013 | 0.000013 | 0.000013 | 0.000017 | 0.000011 | 0.000153 | 0 | 0 |
| pizza50 | `the` | 10 | 761402 | 10 | true | 0.003667 | 0.003665 | 0.003698 | 0.003812 | 0.003642 | 0.003834 | 1596 | 429 |
| pizza50 | `the` | 100 | 761402 | 100 | true | 0.030832 | 0.030773 | 0.031104 | 0.031957 | 0.030676 | 0.032883 | 13433 | 596 |
| synthetic_logs50 | `INFO` | 0 | 294433 | 0 | true | 0.000009 | 0.000009 | 0.000011 | 0.000018 | 0.000008 | 0.000023 | 0 | 0 |
| synthetic_logs50 | `INFO` | 10 | 294433 | 10 | true | 0.001497 | 0.001472 | 0.001599 | 0.001607 | 0.001424 | 0.001608 | 1759 | 502 |
| synthetic_logs50 | `INFO` | 100 | 294433 | 100 | true | 0.009721 | 0.009710 | 0.009767 | 0.009924 | 0.009694 | 0.009932 | 13013 | 617 |
