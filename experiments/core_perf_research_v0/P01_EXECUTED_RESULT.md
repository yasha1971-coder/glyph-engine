# CORE-PERF-P01 — executed performance evidence (2026-11-06)

**Scope:** isolated independently SHA-verified source scan, NOT full GLYPH search/recovery/proof wall time.

[GitHub Actions run 42231485385](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42231485385): **SUCCESS, 3/3 jobs** at commit `642256d1c3740531c13f01016be937bcbd102b74`. Original L9-S golden file Git blob SHA `e0a4c660daa5185dd218e840a5356fceb73c61e5` unchanged. Canonical 100,000,000 B enwik8 SHA256 `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8` independently matched before paired scans.

| Query | Original 2-pass median (s) | New 1-pass median (s) | Speedup |
|---|---:|---:|---:|
| single `e` (8,001,205 hits) | 1.060542 | 0.133839 | **7.924×** |
| two-hit phrase | 0.130089 | 0.078046 | 1.667× |
| absent binary pattern | 0.109359 | 0.057647 | 1.897× |
| NUL byte | 0.103637 | 0.077230 | 1.342× |

Three paired same-runner iterations per profile. Exact physical source bytes read were **200MB→100MB** per query, not guaranteed 2x execution speed. Optimizing the frequent single-byte query additionally uses `bytes.count`, avoiding millions of Python loops; this gain is **not** demonstrated SIMD or whole-Vault acceleration.

Safety: **15,458 independent small binary oracle comparisons**, **13 negative cases rejected**; zero and frequent cases preserve full count and fail-closed completeness state. Mathlib pinned Lean 4.25.0 / commit `1ccd71f89cbbd82ae7d097723ce1722ca7b01c33` checked 3 chunk/carry arithmetic lemmas; this is NOT a proof of SHA256, Python code, signed root/currentness, immutable append semantics, or FM/EF semantics.

**Predictions** recorded in `P01_RESEARCH_PROTOCOL.md` prior to measurement: 50% physical source reads PASS, frequent-byte ≥2x speed PASS (7.924x), ≤1.2x relative runtime for two-hit/empty patterns PASS, exact/corrupt tests PASS. The first [CI 42231298719](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42231298719) failed **before execution** due to shallow checkout history while attempting `git show`; correcting the baseline pin to exact Git blob SHA yielded 3/3 green jobs. Preserve this failed attempt.

**Engineering decision:** keep the original L9 verifier unchanged. This is a new alternative research scanner whose answer is released only after EOF, full source SHA match and stable source fd signature. Next experiment **P02:** batch multiple exact queries in one authenticated source pass, measure single/multi-pattern CPU vs read I/O with 1,2,4,8,16 patterns and enforce complete snapshot/leaf coverage. No new Issue, Draft PR, frozen/main/OVH modification.
