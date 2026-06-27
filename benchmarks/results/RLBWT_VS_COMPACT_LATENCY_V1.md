# RLBWT_VS_COMPACT_LATENCY_V1

Status: measured local benchmark  
Date: 2026-06-26

## Purpose

Compare RLBWT Full Runtime Profile V1 against Compact Runtime Profile V1.

This benchmark measures the tradeoff between:

- runtime size reduction
- query/count/locate latency

Both profiles preserve exact query/count/offset correctness.

## Important boundary

The Compact Runtime path uses the existing C++/Python mixed runtime over raw `bwt.bin`.

The RLBWT Full Runtime path is still a Python prototype over:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`

Therefore this benchmark is not a final production-speed comparison.

It measures current prototype cost.

## Result summary

# RLBWT vs Compact Runtime Latency V1

repeats: 5

| corpus | compact_ratio | rlbwt_ratio | compact_query_avg_sec | compact_locate_avg_sec | compact_total_avg_sec | rlbwt_query_avg_sec | rlbwt_locate_avg_sec | rlbwt_total_avg_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pizza50 | 1.375x | 1.016x | 0.052210 | 0.093391 | 0.145601 | 4.190476 | 4.381695 | 8.572171 |
| xz_cve | 1.594x | 0.815x | 0.001118 | 0.021428 | 0.022546 | 0.029019 | 2.441074 | 2.470094 |
| synthetic_logs50 | 1.375x | 0.526x | 0.051970 | 0.093492 | 0.145462 | 1.132069 | 1.313840 | 2.445909 |

## Interpretation

RLBWT Full Runtime Profile V1 wins on deployed runtime size.

Measured size shift:

- pizza50: Compact 1.375x -> RLBWT 1.016x
- xz_cve: Compact 1.594x -> RLBWT 0.815x
- synthetic_logs50: Compact 1.375x -> RLBWT 0.526x

But current Python RLBWT runtime loses heavily on latency.

Measured total query+locate latency:

- pizza50: Compact 0.145601s -> RLBWT 8.572171s
- xz_cve: Compact 0.022546s -> RLBWT 2.470094s
- synthetic_logs50: Compact 0.145462s -> RLBWT 2.445909s

## Conclusion

The size problem is no longer the primary blocker.

The new primary blocker is compressed runtime speed.

RLBWT Full Runtime Profile V1 proves compressed query/count/locate correctness.

The next required engineering step is a C++ RLBWT runtime core.

## Next technical target

Implement C++ RLBWT runtime primitives:

1. Load RLB1 container.
2. Load RLBWT rank blocks.
3. Implement symbol_at(pos).
4. Implement rank(c,pos).
5. Implement LF.
6. Implement backward search.
7. Implement locate over LOC1 sampled SA.
8. Benchmark against Python RLBWT runtime and Compact Runtime V1.

Production goal:

Preserve RLBWT Full Runtime size profile while reducing query+locate latency by at least 10x.
