# LATENCY SCALING LAW V1

Status:
EXPERIMENTAL LAW

Runtime:
query_fm_server_v1

Index format:
FMBINv2

Retrieval mode:
persistent resident FM

Query mode:
steady-state exact substring retrieval

Checkpoint step:
256

## Objective

Determine whether persistent FM exact substring retrieval latency materially degrades as corpus size increases.

## Method

For each corpus size:

- build canonical FMBINv2 index
- warm runtime state
- execute steady-state latency benchmark
- execute repeatability benchmark
- evaluate p95 stability

All measurements performed on resident FM indexes.

## Scaling Points

| Corpus | p50_ms | p95_ms | p99_ms | Verdict |
|---|---:|---:|---:|---|
| 512MB | ~0.006-0.009 | ~0.007-0.010 | ~0.008-0.011 | LAW_CONFIRMED |
| 1GB | ~0.006-0.009 | ~0.007-0.010 | ~0.008-0.011 | LAW_CONFIRMED |
| 2GB | ~0.006-0.009 | ~0.007-0.010 | ~0.008-0.011 | LAW_CONFIRMED |

## Repeatability Results

512MB:
p95 ratio = 1.440

1GB:
p95 ratio = 1.493

2GB:
p95 ratio = 1.415

Acceptance threshold:
p95 max/min < 1.5x

## Main Observation

Latency remains within the same microsecond-scale regime from 512MB through 2GB.

No major latency collapse was observed.

## Interpretation

Current dominant retrieval cost appears to be:

- Occ traversal
- CPU/cache behavior
- runtime scheduling noise

NOT:

- disk IO
- page residency collapse
- large-scale memory traversal failure

## Important Runtime Observation

Two recurring latency modes were repeatedly observed:

FAST MODE:
~0.006ms p50

NORMAL MODE:
~0.009ms p50

Possible causes:

- scheduler state
- CPU frequency behavior
- cache residency effects
- transient runtime noise

This does not invalidate the scaling law.

## Current Law Formulation

LATENCY_SCALING_LAW_V1:

For resident FMBINv2 indexes with checkpoint_step=256,
GLYPH preserves microsecond-scale steady-state exact substring retrieval latency through at least 2GB corpus scale.

## Next Frontier

Potential future degradation sources:

- LLC pressure
- TLB pressure
- NUMA behavior
- page-cache fragmentation
- memory bandwidth saturation

Suggested next scaling point:

4GB

## Important Limitation

This law currently applies ONLY to:

- resident indexes
- persistent runtime
- steady-state retrieval
- warm memory state

It does NOT yet describe:

- cold-start latency
- distributed retrieval
- HTTP/runtime overhead
- shard federation
- network retrieval
