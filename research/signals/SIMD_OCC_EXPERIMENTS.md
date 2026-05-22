# SIMD OCC EXPERIMENTS

Goal:

Accelerate Occ().

Candidates:

1.

AVX2 byte compare

Idea:

load 32B

compare byte lane

popcount mask

2.

AVX512 compare

64B lanes

3.

checkpoint prefetch

4.

branch reduction

5.

hybrid scalar/SIMD threshold

Measure:

Occ benchmark V1 only.

Metrics:

p50
p95
p99

queries/sec

Rules:

No merge without benchmark gain.

Correctness mandatory.

Determinism unchanged.
