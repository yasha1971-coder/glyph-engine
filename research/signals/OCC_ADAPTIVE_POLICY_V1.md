# OCC ADAPTIVE POLICY V1

Purpose:

Define when GLYPH should use scalar Occ and when it should use AVX2 Occ.

Measured basis:

OCC_STEP_BENCH_V1

Machine:

AMD EPYC 4344P

Observed breakpoint:

scan_len around 64 bytes.

Measured interpretation:

scan_len <= 32

scalar and AVX2 are effectively tied

scan_len 64

AVX2 begins reducing tail latency

scan_len >= 128

AVX2 becomes clearly useful

Policy V1:

if scan_len < 64

use scalar Occ

else

use AVX2 Occ when available

Rationale:

Current GLYPH latency layout keeps avg_scan_bytes around 28.

Therefore scalar remains optimal or equal for the common short-scan path.

AVX2 is valuable for longer scan paths created by larger checkpoint_step values.

This avoids forcing SIMD overhead onto tiny scans.

Architecture implication:

SIMD is not a global replacement for scalar Occ.

SIMD is a conditional execution path.

Future layout profiles:

LATENCY

checkpoint_step around 32

expected scan below 64

default Occ path:

scalar

BALANCED

checkpoint_step around 128

expected scan around breakeven

default Occ path:

adaptive scalar/AVX2

COMPACT

checkpoint_step 256+

expected scan above breakeven

default Occ path:

AVX2 preferred

Future branch:

bit-plane Occ layout

This is separate from byte-comparison AVX2.

Bit-plane layout may become GLYPH_LAYOUT_V2.

Current policy applies only to byte-layout BWT.
