# RLBWT_PERSISTENT_CPP_SERVER_V1_BENCH

Status: measured local benchmark  
Date: 2026-06-27

## Purpose

Benchmark persistent C++ RLBWT full query+locate server.

Server binary:

    rlbwt_full_query_locate_server_v1

Runtime files are loaded once:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

Then repeated `query_hex` requests are sent through stdin.

This measures warm repeated query+locate latency after runtime load.

## Interpretation note

Persistent server removes process startup and repeated runtime loading.

For single-hit queries, warm latency becomes extremely small.

For multi-hit locate-heavy queries, such as XZ CVE with 38 offsets and 5531 LF steps, warm latency is dominated by actual locate work rather than process/loading overhead.

# RLBWT Persistent C++ Server V1 Benchmark

one-shot repeats: 7

server warm repeats: 100

| corpus | one_shot_avg_sec | server_first_response_sec | server_warm_avg_sec | server_warm_min_sec | one_shot / warm_server |
|---|---:|---:|---:|---:|---:|
| pizza50 | 0.068368 | 0.056483 | 0.000228 | 0.000225 | 299.26x |
| xz_cve | 0.008429 | 0.009780 | 0.008944 | 0.008857 | 0.94x |
| synthetic_logs50 | 0.045934 | 0.036795 | 0.000135 | 0.000130 | 340.79x |
