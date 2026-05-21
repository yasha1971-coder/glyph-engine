GLYPH BENCHMARK SCOPES

Purpose

Define benchmark scopes so results are comparable across machines,
commits, corpora, and runtime contracts.

Scopes

1. count_single

One process.
One FM query.
Includes process startup.
Measures CLI overhead + artifact load + query.

2. count_persistent

One persistent FM server.
Many queries.
Excludes repeated startup.
Measures core FM query path.

3. count_batch_http

One HTTP request.
Multiple exact patterns.
Measures HTTP bridge + persistent backend + batch protocol.

4. verified_query

Manifest verification + FM query.
Measures integrity wrapper cost.

5. locate_verify

FM count + locate_backend_v2 + corpus slice verification.
Measures exact byte-offset recovery.

6. segmented_fanout

Multiple shards.
Same query sent across shard set.
Measures fan-out overhead and aggregation cost.

7. cold_start

Drop/reduce cache if possible.
Start backend from cold filesystem state.
Measures artifact load and mmap/page-cache behavior.

8. warm_query

Repeated query after backend is resident.
Measures stable p50/p95/p99 retrieval latency.

Required fields per benchmark

commit
machine_id
cpu_model
cpu_arch
kernel
compiler
simd_capability
corpus_name
corpus_size
artifact_versions
query_set
scope
runs
p50_ms
p95_ms
p99_ms
mean_ms
min_ms
max_ms
rss_mb
notes

Principles

Correctness before speed.

Each benchmark must state its scope.

Never compare different scopes as if they are the same.

Warm latency and cold startup are separate metrics.

Batch latency and single-query latency are separate metrics.

Verified query and raw FM query are separate metrics.

Segmented fan-out must report shard count.

Machine spec is part of the result.

Runtime capability is part of the result.

Silent benchmark drift is worse than slower numbers.
