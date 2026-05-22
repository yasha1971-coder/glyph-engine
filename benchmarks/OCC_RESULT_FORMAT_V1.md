# OCC RESULT FORMAT V1

Metadata:

machine_spec

dataset

commit

compiler

build_flags

occ_variant

scalar

simd_avx2

simd_avx512

hybrid

Metrics:

p50_ns

p95_ns

p99_ns

throughput_queries_sec

throughput_gbps

rss_mb

cold_warm_mode

correctness

PASS

FAIL

Output:

JSON preferred.

Rules:

No benchmark accepted without commit hash.

No benchmark accepted without machine spec.

No benchmark accepted without correctness PASS.

No benchmark accepted without protocol version.

Benchmark result format changes require version bump.
