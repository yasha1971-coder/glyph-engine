# RLBWT_SERVER_MULTIQUERY_V1

Status: measured local benchmark  
Date: 2026-06-27

## Purpose

Benchmark persistent C++ RLBWT full query+locate server on multiple warm workload types.

Server:

    rlbwt_full_query_locate_server_v1

Runtime files are loaded once:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

Workloads:

- single-hit query
- absent/miss query
- multi-hit query
- mixed workload

Metrics:

- avg
- p50
- p95
- p99
- min/max
- total LF steps
- max LF steps

## Interpretation

Miss queries are extremely fast when the FM interval collapses before locate.

Single-hit queries are sub-millisecond in warm server mode.

Multi-hit queries are dominated by offset recovery and LF steps.

# RLBWT_SERVER_MULTIQUERY_V1

server warm repeats per workload: 200

| corpus | workload | expected_count | avg_sec | p50_sec | p95_sec | p99_sec | min_sec | max_sec | total_steps | max_steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pizza50 | hit_single | 1 | 0.000223 | 0.000221 | 0.000234 | 0.000248 | 0.000220 | 0.000273 | 54 | 54 |
| pizza50 | miss_absent | 0 | 0.000020 | 0.000021 | 0.000021 | 0.000023 | 0.000020 | 0.000025 | 0 | 0 |
| pizza50 | mixed | -1 | 0.000122 | 0.000220 | 0.000224 | 0.000226 | 0.000020 | 0.000227 | -1 | -1 |
| xz_cve | multi_hit | 38 | 0.008843 | 0.008849 | 0.008911 | 0.009035 | 0.008728 | 0.009072 | 5531 | 473 |
| xz_cve | miss_absent | 0 | 0.000013 | 0.000013 | 0.000013 | 0.000018 | 0.000013 | 0.000023 | 0 | 0 |
| xz_cve | mixed | -1 | 0.004439 | 0.008846 | 0.008862 | 0.008927 | 0.000017 | 0.009158 | -1 | -1 |
| synthetic_logs50 | hit_single | 1 | 0.000134 | 0.000133 | 0.000144 | 0.000153 | 0.000131 | 0.000178 | 37 | 37 |
| synthetic_logs50 | miss_absent | 0 | 0.000040 | 0.000040 | 0.000040 | 0.000044 | 0.000040 | 0.000073 | 0 | 0 |
| synthetic_logs50 | mixed | -1 | 0.000087 | 0.000132 | 0.000134 | 0.000136 | 0.000040 | 0.000142 | -1 | -1 |
