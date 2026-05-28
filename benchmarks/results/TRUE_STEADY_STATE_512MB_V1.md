# TRUE STEADY STATE 512MB V1

Runtime:
query_fm_server_v1

Harness:
tools/steady_state_latency_bench_v1.py

Corpus:
data_4gb/test_512mb.bin

Corpus size:
536,870,912 bytes

FM format:
FMBINv2

Checkpoint step:
256

Query file:
bench_queries/basic_1000_v1.txt

Warmup:
50

Results:

avg_ms=0.006191
p50_ms=0.006080
p95_ms=0.007203
p99_ms=0.007633

min_ms=0.004079
max_ms=0.011280

qps=161531.735

Interpretation:

This benchmark isolates true steady-state FM traversal latency.

Startup/load cost is excluded.

Key finding:

FM traversal itself is extremely cheap.

Most retrieval latency previously observed came from:
- process startup
- FM/BWT loading
- runtime lifecycle overhead
- orchestration path

Architectural conclusion:

Persistent resident runtime is the correct retrieval architecture.

Retrieval physics decomposition:

single CLI query:
~900ms scale

batch amortized:
~2.75ms/query

true steady-state traversal:
~0.006ms/query

Result:

GLYPH retrieval core operates in microsecond-scale steady-state latency.
