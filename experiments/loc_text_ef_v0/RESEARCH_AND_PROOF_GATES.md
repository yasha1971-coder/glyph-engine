# CORE-LF-BOUND-1 — research branch, not a LOC2 replacement

**Date:** 2026-11-05
**Status:** hypothesis + executable small exhaustive evidence; enwik8 physical gate separately required.
**Frozen source:** \`rlbwt-binary-safe-v2\` at \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`.
**Owner strategic veto:** no main, frozen manifest, ABI, runtime, OVH or release mutation permitted by this research.

## Original GLYPH target preserved

The goal is a **compact, exact, independently verifiable and byte-restorable search state**, with stable corpus/document identity and fail-closed P/R/O replay. Optimizing a sampled SA file alone is not success. Existing LOC1/LOC2, V1 trust, Composition, experimental field masks and ACEAPEX must remain independent until the explicit proof gate is passed.

## Observation / prediction

Original experimental LOC2 chooses every \`s\`-th **SA row**. A row sample of size \`ceil(N/s)\` occupies 4-byte offsets on canonical 100MB enwik8 (actual LOC2 3,125,036 B), but does not imply that *every* row reaches one of those samples after \`<=s-1\` LF transitions. At least one verified historical V2 workload showed a 391-step path at step 128; do not extrapolate that to universal worst-case.

**Known alternative, not our new mathematical invention:** sample values whose **text position** \`SA[row] % s == 0\`. Because correctly implemented LF maps \`SA[row]\` to \`(SA[row]-1) mod N\`, for each original SA value \`x\`, exactly \`x % s\` LF steps suffice to hit a multiple of \`s\`. Therefore maximal locate LF distance \`<= s-1\`; this is a standard text-order sampled FM-index technique. A physical implementation also requires membership testing and recovering the sampled SA value for the reached row. Citations to existing technique:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5956352/
- https://doi.org/10.1145/3750729
- https://arxiv.org/abs/1705.10382

### Proposed combined representation (research-only)

Store text-position samples *sorted in SA-row order* as (1) an Elias–Fano set of sampled row IDs, (2) quotient \`SA[row]/s\` instead of the full 32-bit SA value, and (3) a **physically accounted** select accelerator. This retains exactly \`ceil(N/s)\` sampled values but changes their location to guarantee the LF bound. For a given sample row \`r\`, recover \`SA[r] = quotient*s\`, and for another row after \`d\` LF steps, \`SA[original] = (SA[r] + d) mod N\`.

No claim of conceptual priority: text-order sampling, Elias–Fano and quotient bitpacking all have prior art. Only their actual engineering combination under GLYPH's manifest/trust contract, if it proves effective, can be a legitimate GLYPH contribution.

### Frozen enwik8 100MB size prediction, **not yet observed file**

For \`N=100,000,001\`, \`s=128\`, \`m=floor((N-1)/s)+1=781,251\`:

| Item | Derived quantity | Predicted bytes |
|---|---|---:|
| Elias–Fano low bits | \`l=floor(log2(floor(N/m)))=6\` | 585,939 |
| Elias–Fano high bitvector | \`((N-1)>>6)+m+1=2,343,752\` bits | 292,969 |
| Quotient array | \`ceil(log2 m)=20\` bits/value | 1,953,128 |
| Select1 checkpoints | every 256 samples, 32-bit physical offset | 12,208 |
| Research header | version/geometry/section sizes | 80 |
| **Total proposed experimental locate file** | | **2,844,324** |
| Actual historical LOC2 | same 128-step parameter; different sampling invariant | 3,125,036 |
| **Potential additional saving** | only if physically reproduced | **280,712** |

The 20-bit quotient is a **key saving**: the sample values are multiples of 128 and need not carry seven guaranteed zero bits. A row membership data structure is unavoidable; omitting it creates a dishonest storage claim. This estimate includes an *initial* select accelerator but excludes runtime code, metadata trust, any extra index/caches or independent verifier. Checkpoint offsets are u32 **only while** high bitvector positions fit 32 bits; a 64-bit format path must be defined before scaling. For non-powers of two use exact integer division and enforce canonical geometry.

**Target gain is not speed in every query:** count uses unchanged rank; locate on some queries may pay for Elias–Fano select and binary search. P95 and p99 versus original LOC2 must be measured, not predicted; the upper bound is in LF steps, not constant wall time.

## Executable falsifier and gates

1. **CORE-LF-A — exhaustive small binary:** include NUL, 0xFF, overlaps, identical bytes and empty text, binary-safe sentinel-first SA and LF derived independently from BWT C/rank. For all rows and steps verify exact SA recovery, \`LF<=s-1\`, and distinguish the historical row-sampled counterexamples. At least one true historical sampling violation is required to show the test is not vacuous.
2. **CORE-LF-B — actual enwik8 physical file:** regenerate public canonical enwik8 source and frozen SA SHA256 \`a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727\`; physically pack EF rows + quotients + counted checkpoints. Check 781,251/781,251 samples against canonical SA and reject malformed/truncated geometry. Record actual file bytes and SHA. Do not call a computed estimate a measurement.
3. **CORE-LF-C — actual GLYPH rank/locate parity:** bind unchanged RLB2/RLR2 and new candidate sampler in a separate runtime adapter. Independent direct-byte oracle checks full \`count\`/\`locate\` including the historical \`max_offsets\` ambiguity. Prove all returned coordinates, not only count; normalise canonical order before claiming O3/API compatibility.
4. **CORE-LF-D — formal Lean/Mathlib model:** state LF permutation/sentinel assumptions and the \`SA[row]%s\` stopping bound, machine check using pinned Lean and Mathlib. The mathematical paper proof does **NOT** equal Lean-checked until a compiler actually accepts it. No \`sorry\` or unsound axiom placeholders. Lean/Mathlib docs: https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Nat/ModEq .
5. **CORE-LF-E — complete proof and retained bytes:** P/R/O query evidence, authenticated byte extraction, independently pinned manifest, hostile mutation and stale generation, independent replay without build-time SA/corpus, verified storage ratio including payload and proof, CPU-native cold/warm p50/p95/p99. Failing any mandatory dependency -> UNTRUSTED, never false PROVEN_EMPTY.

The newly public \`openai/math\` https://github.com/openai/math is *research data* with varied verification stages; it is not a general theorem oracle. Borrow its rigorous verification workflow, not unverified claims of proved GLYPH theorems.

**No release promotion** until B/C/D/E are all executed and the owner explicitly approves the required architectural mutation. Distinguish execution statuses for every gate.
