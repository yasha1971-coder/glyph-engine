LATENCY_REGIME_VALIDATION_V1

Status: VALIDATED
Date: 2026-05-31

Objective

Determine whether exact FM-index retrieval latency changes as corpus size increases from 512MB to 2GB.

This experiment evaluates retrieval behavior only.

It does not evaluate:

* index build time
* memory footprint scaling
* startup/loading latency
* SA32/SA64 migration effects

The goal is to isolate steady-state query latency.

⸻

Test Environment

Hardware:

* AMD EPYC 4344P
* DDR5 memory
* Linux
* Persistent resident process

Software:

* GLYPH FM-index backend
* FMBINv2 artifacts
* checkpoint_step = 256

Query set:

* 1000 repeated exact substring lookups
* identical query set across all corpus sizes

Measurement method:

* warmup = 50 queries
* steady-state measurements only
* raw latency samples collected
* p50/p95/p99 computed from raw observations

⸻

Corpus Sizes

Corpus

512MB

1GB

2GB

⸻

Results

Corpus	p50 (ms)	p95 (ms)	p99 (ms)
512MB	0.009038	0.010271	0.010939
1GB	0.008957	0.010209	0.010740
2GB	0.008987	0.010248	0.010688

⸻

Variation Across Sizes

Observed spread:

p50:

* min = 0.008957 ms
* max = 0.009038 ms
* delta = 0.000081 ms

p95:

* min = 0.010209 ms
* max = 0.010271 ms
* delta = 0.000062 ms

p99:

* min = 0.010688 ms
* max = 0.010939 ms
* delta = 0.000251 ms

⸻

Distribution Analysis

Raw latency distributions were collected.

Result:

* no discrete latency modes observed
* no evidence of stable bi-modal behavior
* dominant latency peak centered near 0.009 ms
* low-latency observations (~0.006 ms) appear as tail events rather than a separate operating mode

Observed behavior is consistent with a single latency regime.

⸻

Interpretation

Across a 4× corpus growth:

512MB → 1GB → 2GB

steady-state retrieval latency remained within the same measured microsecond regime.

Within benchmark resolution:

* no measurable latency degradation was observed
* corpus growth did not materially affect p50, p95, or p99 latency

⸻

Current Scaling Boundary

The next observed limitation is not retrieval latency.

The current engineering boundary is representation width:

* SA32 practical limit
* corpus size growth beyond current safe range
* future SA64 or sharding decisions

At present, retrieval latency does not appear to be the dominant scaling constraint.

⸻

Conclusion

Observed result:

Steady-state exact retrieval latency remained effectively constant across 512MB, 1GB, and 2GB corpora.

Current evidence suggests that the first scaling boundary encountered by GLYPH is representation width (SA32), not retrieval latency.
