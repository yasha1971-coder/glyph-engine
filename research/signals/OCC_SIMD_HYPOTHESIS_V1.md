OCC SIMD HYPOTHESIS V1

Baseline scalar:
EPYC4344P
p50 20ns
p95 30ns
p99 30ns

Observation:

manual unroll reduces tail latency

Hypothesis:

AVX2 compare+movemask+popcnt
may reduce p50

candidate:

_loadu_si256
_cmpeq_epi8
_movemask_epi8
_popcnt

goal:

p50 < 20ns
