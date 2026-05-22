# SIMD OCC PLAN

Goal:

Accelerate FM rank/Occ layer.

Motivation:

Correctness and contracts are stabilized.

Next leverage point:

Occ()

Current:

scalar implementation

Future:

scalar
↓

AVX2
↓

AVX512

Benchmark isolation requirements:

measure:

- Occ only
- rank only

exclude:

- mmap startup
- HTTP
- IPC
- backend startup
- shard orchestration

Benchmark outputs:

p50
p95
p99

Machine profile mandatory:

- CPU
- compiler
- SIMD capability
- corpus size

Success metric:

lower rank latency

Future dispatch layer:

capability probe
↓

runtime dispatch
↓

scalar / avx2 / avx512

Principle:

benchmark = contract

performance follows correctness
