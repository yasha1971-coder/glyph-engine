# OCC BENCH V1

Purpose:

Measure Occ()/rank layer independently.

Scope:

INCLUDE:

- rank()
- Occ()
- checkpoint lookup

EXCLUDE:

- mmap
- manifest verification
- runtime gate
- HTTP
- IPC
- startup

Metrics:

throughput:
queries/sec

latency:
p50
p95
p99

Corpora:

mini
1GB
future: 8GB segmented

Machine spec required.

Output format:

machine
compiler
flags
dataset
query_count

rank_ns_p50
rank_ns_p95
rank_ns_p99

occ_ns_p50
occ_ns_p95
occ_ns_p99

Principle:

measure subsystem

not whole engine
