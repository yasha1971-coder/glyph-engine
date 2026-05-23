# BACKWARD SEARCH BENCH V1

Purpose:

Measure real FM query pipeline below CLI/HTTP.

Scope includes:

- pattern bytes
- backward_search()
- adaptive Occ()
- interval result
- count result

Scope excludes:

- HTTP
- JSON serialization
- process startup
- manifest verification
- runtime gate

Variants:

scalar Occ

adaptive Occ

Pattern lengths:

4 bytes
8 bytes
16 bytes
32 bytes

Metrics:

p50_ns
p95_ns
p99_ns
queries_per_sec

Dataset:

mini first

Future:

1GB HDFS
segmented shards

Goal:

Move from Occ microbenchmark to query pipeline benchmark.

Principle:

micro law must be validated at pipeline level
