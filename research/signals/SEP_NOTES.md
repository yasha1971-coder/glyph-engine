# SEP NOTES

Reference source:https://github.com/nietras/Sep

Purpose:

Reference system for benchmark organization,
scope separation,
machine specification discipline,
and future SIMD-path benchmarking methodology.

Observed useful patterns:

1. Benchmark scope separation

Benchmark layers should remain isolated.

Example principle:

core algorithm benchmark

!=

system benchmark

!=

startup benchmark

!=

IPC benchmark

!=

HTTP benchmark

Interpretation for GLYPH:

Occ/rank benchmarks should be isolated from:

- mmap startup
- backend startup
- HTTP layer
- IPC overhead
- shard orchestration

Benchmark signal becomes cleaner.

2. Machine specification discipline

Benchmarks without machine context lose value.

Machine profile should always include:

- CPU model
- core/thread count
- RAM
- compiler
- SIMD capability
- kernel
- corpus size
- shard count

Interpretation:

benchmark = contract

3. SIMD direction

Future GLYPH acceleration direction:

Occ()

↓

scalar baseline

↓

AVX2

↓

AVX512

Goal:

reduce rank latency

reduce shard fanout overhead

increase deterministic retrieval throughput

Future GLYPH work:

- SIMD Occ layer
- isolated Occ benchmark
- benchmark reproducibility discipline

Strategic interpretation:

benchmark != marketing

benchmark = engineering contract
