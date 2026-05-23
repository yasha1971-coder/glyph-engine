# OCC POLICY UPDATE V2

Previous threshold:

64 bytes

Measured result:

EPYC4344P

adaptive threshold 64:

adaptive followed scalar path

adaptive threshold 32:

adaptive matched SIMD tail latency

Measured policy:

scan_len < 32

scalar

scan_len >= 32

AVX2

Reason:

Current GLYPH avg_scan_bytes ~28

Threshold 32 captures long-tail scans

without forcing SIMD overhead

Status:

verified

Machine:

AMD EPYC 4344P

Benchmark:

OCC_BENCH_V1

Result:

Policy V2 supersedes V1

Future:

threshold may become hardware-profile dependent

EPYC

consumer CPU

ARM

Future adaptive branch:

auto threshold calibration
