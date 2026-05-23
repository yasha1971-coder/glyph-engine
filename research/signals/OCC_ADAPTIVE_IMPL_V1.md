# OCC ADAPTATIVE IMPLEMENTATION V1

Goal:

Move adaptive Occ policy from documentation into executable backend logic.

Current policy:

scan_len < 64

scalar

scan_len >= 64

AVX2

Execution model:

occ_core()

becomes:

occ_core_adaptive()

Pseudo:

if scan_length < adaptive_threshold

return occ_scalar()

else

return occ_avx2()

Runtime threshold:

64

Future:

threshold may become configurable.

Possible future manifest:

occ_policy:

scalar

adaptive

avx2

Possible runtime capability:

CPU AVX2 absent

force scalar

CPU AVX2 available

adaptive allowed

Future tuning:

threshold auto-learning

hardware profile aware

EPYC profile

laptop profile

ARM profile

Constraint:

adaptive dispatch cost must remain below gain.

Dispatch overhead must be measured.

Measurement surface:

bench_occ_dispatch_v1

Expected effect:

keep latency optimal

without forcing SIMD overhead

Architecture result:

GLYPH becomes policy driven

not implementation driven
