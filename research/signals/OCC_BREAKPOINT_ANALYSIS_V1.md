# OCC BREAKPOINT ANALYSIS V1

Date:

2026-05-22

Machine:

AMD EPYC 4344P

Benchmark:

OCC_STEP_BENCH_V1

Dataset:

mini BWT

Iterations:

100000

Measured result:

scan_len 16

scalar p50 20ns
simd_avx2 p50 20ns

scan_len 32

scalar p50 20ns
simd_avx2 p50 20ns

scan_len 64

scalar p50 20ns
simd_avx2 p50 20ns

scalar p95 30ns
simd_avx2 p95 20ns

scan_len 128

scalar p50 30ns
simd_avx2 p50 20ns

scan_len 256

scalar p50 30ns
simd_avx2 p50 20ns

scan_len 512

scalar p50 40ns
simd_avx2 p50 20ns

scan_len 1024

scalar p50 50ns
simd_avx2 p50 30ns

Core finding:

AVX2 byte-compare Occ becomes clearly useful around scan_len 64-128 bytes.

For scan_len <= 32 bytes, scalar and AVX2 are effectively tied.

Interpretation:

Current GLYPH mini FM layout has avg_scan_bytes about 28.

Therefore, current checkpoint density already keeps Occ scans below the main AVX2 breakeven point.

This explains why AVX2 did not significantly improve p50 in real Occ benchmark.

Important conclusion:

SIMD was not useless.

SIMD revealed that the current checkpoint layout is already latency-oriented.

Layout implication:

checkpoint_step 32

latency profile

expected scan below 64 bytes

SIMD optional

checkpoint_step 128

balanced profile

SIMD begins to matter

checkpoint_step 256+

compact / memory-saving profile

SIMD recommended or required

Future direction:

Do not jump directly to AVX512.

Next real experiment:

measure checkpoint_step vs:

- FM size
- avg_scan_bytes
- scalar latency
- AVX2 latency
- memory footprint

Strategic meaning:

checkpoint_step is not only a builder parameter.

checkpoint_step is part of a future Layout Contract.

Possible future layout profiles:

latency

balanced

compact

Byte-layout ceiling:

Current SIMD approach uses byte comparison:

BWT[i] == symbol

Real SIMD-native FM layout may require bit-plane / strided representation.

But bit-plane layout is deferred.

First exhaust byte-layout behavior.
