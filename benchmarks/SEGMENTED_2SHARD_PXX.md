# GLYPH Segmented 2-Shard Benchmark

## Benchmark

Tool:

    benchmarks/segmented_fanout_v1.py

Mode:

    sequential_fanout_over_persistent_cpp_backends

Corpus:

    HDFS 1GB split into 2 shards

Shard layout:

    shard0: first 512 MiB
    shard1: second 512 MiB

Query set:

    bench_1gb/queries.txt

Query count:

    100

Warm runs:

    3

Warm measurements:

    300

---

## Single-index baseline

Reference benchmark:

    benchmarks/HDFS_1GB_PERSISTENT_PXX.md

Single 1GB persistent backend:

    warm p50:  ~0.0098 ms
    warm p95:  ~0.0104 ms
    warm p99:  ~0.0105 ms

---

## 2-shard segmented result

Two persistent backends queried sequentially.

Startup:

    p50: ~1910.8 ms per shard

Cold queries:

    p50: ~0.0270 ms
    p95: ~0.0376 ms
    p99: ~0.0819 ms

Warm queries:

    min:  ~0.0192 ms
    p50:  ~0.0207 ms
    p95:  ~0.0228 ms
    p99:  ~0.0239 ms
    max:  ~0.0291 ms
    mean: ~0.0210 ms

---

## Sample fan-out response

Query:

    blk_-100000266894974466

Responses:

    shard0:
        count = 31

    shard1:
        count = 0

Total:

    31

---

## Interpretation

This benchmark measures sequential fan-out across two persistent
FM backends.

Observed behavior:

    single backend p50:
        ~0.010 ms

    two-shard fan-out p50:
        ~0.021 ms

Observed scaling:

    ~2.1× latency increase

This is consistent with sequential querying over two resident indexes.

Important:

No explosive tail-latency behavior was observed.

Tail stability:

    p99 / p50 ≈ 1.15

This indicates stable warm-query behavior even under segmented fan-out.

---

## What this benchmark does NOT measure

- parallel shard querying
- shard overlap handling
- cross-shard locate merge cost
- HTTP/network overhead
- distributed multi-machine fan-out
- shard balancing strategies

---

## Machine

See:

    benchmarks/MACHINE_SPEC.md
