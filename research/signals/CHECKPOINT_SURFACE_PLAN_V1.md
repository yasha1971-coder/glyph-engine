# CHECKPOINT SURFACE PLAN V1

Goal:

Measure checkpoint_step tradeoffs.

Parameters:

checkpoint_step:

16
32
64
128
256
512
1024

Measure:

fm_size_bytes

avg_scan_bytes

scalar_occ_p50_ns

simd_occ_p50_ns

simd_gain_ratio

memory_overhead

Questions:

Q1

Where is latency optimum?

Q2

Where does SIMD begin paying off?

Q3

How expensive is checkpoint density?

Q4

Can layout profiles become stable contracts?

Future Layout Contract:

LATENCY

checkpoint optimized for small scan

BALANCED

checkpoint optimized for mixed workloads

COMPACT

checkpoint optimized for memory

Output format:

JSON deterministic.

Future artifact:

GLYPH_LAYOUT_CONTRACT_V1
