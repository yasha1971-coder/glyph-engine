# PERSISTENT RETRIEVAL ROADMAP V1

Status:

Validated on Windows/MSVC using enwik8.

Core finding:

FM traversal is not the dominant bottleneck in current one-shot CLI query mode.

Observed:

Single CLI query:
~906-949 ms/query

Persistent/batch stdin mode:
100 queries in 764.529 ms
~7.645 ms/query including process startup/load

Manual persistent queries:
visually instant

Conclusion:

The main bottleneck is process/artifact lifecycle:

- process startup
- artifact open/load
- FM/BWT setup
- per-query CLI execution

Not primarily:

- backward search
- match count
- pattern frequency

Architecture direction:

1. Persistent FM daemon
2. Hot index residency
3. Batch query protocol
4. Binary request/response protocol
5. Zero-copy query path
6. mmap-aware artifact loading
7. Persistent latency benchmark
8. p50/p95/p99 measurement
9. Windows/Linux parity tests

Important distinction:

query_fm_v1 = one-shot CLI path
query_fm_server_v1 = stdin/stdout persistent FM engine
glyph_http_server.py = experimental HTTP layer, not raw FM latency

Next required benchmark:

PERSISTENT_LATENCY_BENCH_V1

Measure:
- same pattern repeated
- mixed patterns
- absent patterns
- short/long patterns
- p50/p95/p99
- throughput queries/sec

Target corpora:
- examples/mini
- enwik8
- Silesia
- enwik9 step=256
