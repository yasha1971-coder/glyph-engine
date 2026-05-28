# TRUE STEADY STATE REPEATABILITY V1

Purpose:

Prove whether the observed ~6 microsecond persistent FM steady-state latency is a stable system law or a one-off observation.

Main number under test:

persistent FM steady-state latency

Current observed value:

~0.006 ms/query
~6 microseconds/query

Corpus:

data_4gb/test_512mb.bin

FM format:

FMBINv2

Checkpoint step:

256

Known result:

avg_ms=0.006191
p50_ms=0.006080
p95_ms=0.007203
p99_ms=0.007633
qps=161531.735

## Hypothesis

The ~6us latency is stable because:

- FM/BWT artifacts are resident in RAM/page cache
- disk IO is not in the hot path
- each query touches a small number of pages
- Occ scan cost is small relative to process/runtime overhead
- resident persistent runtime removes process startup/load cost

## Questions

Only three questions are in scope.

### 1. Is ~6us stable across repeated runs?

Plan:

- 10 consecutive runs
- same corpus
- same query file
- same runtime
- record avg/p50/p95/p99/max/qps per run

Success signal:

- p50/p95/p99 remain within a small band
- variance should remain below 2x unless system noise explains it

### 2. Does latency drift over time?

Plan:

- run now
- run after 5 minutes
- run after 30 minutes
- run after 1 hour

Goal:

Detect thermal drift, scheduler noise, page cache eviction, or runtime instability.

### 3. How does latency scale with corpus size?

Plan:

- 512MB known baseline
- 1GB
- 2GB
- optional 4GB

Goal:

Find the corpus-size threshold where steady-state latency begins to degrade.

## Out of scope

Do not benchmark:
- HTTP
- RPC
- dashboards
- distributed shards
- async layers
- zero-copy protocol
- UI
- external orchestration

These are later layers.

## Main rule

Do not inflate benchmarks.

The benchmark exists to validate one number:

persistent FM steady-state latency

Everything else is context.

## Strategic meaning

If repeatable, GLYPH has a microsecond-scale exact substring retrieval core.

This shifts the engineering focus from FM traversal optimization to:

- persistent daemon
- hot index residency
- request protocol
- lifecycle elimination
- corpus-size latency scaling
