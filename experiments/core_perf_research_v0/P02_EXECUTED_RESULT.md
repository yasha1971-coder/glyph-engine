# CORE-PERF-P02 — independently checked batch exact-source scan

**Evidence cut:** 2026-11-06. Research only; no GLYPH L9/L10 release change.

Source branch: `codex/core-l10-append-replay-20261106` (existing [Draft PR #29](https://github.com/yasha1971-coder/glyph-engine/pull/29) / [Issue #28](https://github.com/yasha1971-coder/glyph-engine/issues/28)).
Initial executable [GitHub Actions run 42276635923](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42276635923): **SUCCESS 2/2** at `b1a485bcf82ba984ca592a766b402c60931eaa4f`. A separate formal Mathlib job was added later and must have its **own** success evidence; do not project this initial PASS onto the new proof code.

## Experiment and why it matters

Comparison is NOT the historical two-pass reference. It compares the original **P01 one-pass per query** baseline (`one_pass_scan.py`, Git blob `56245f51a2dd0c4b17fb018d277b5222096448f9`) to **P02 batched one-pass for 1/2/4/8/16 binary patterns** (`batch_pass_v0.py`). The earlier `core_l9_scale_v0/verifier.py` is still pinned to Git blob `e0a4c660daa5185dd218e840a5356fceb73c61e5` and unchanged.

For every immutable externally SHA-pinned source:
- sequential baseline performs Q full source reads and Q full source SHA checks, one per pattern;
- P02 performs **one** read and one full SHA check, while testing all Q patterns on each in-memory block;
- all results withheld until full EOF, source SHA equality and unchanged fd geometry; result `PROVEN_EMPTY` only for the complete pinned *single source*; a count with >256 offsets returns `UNTRUSTED / LIMITED_COMPLETENESS`;
- **mixed-length carry invariant**: use maximum pattern length minus one as carry; discard a shorter-pattern hit if its ending byte was already in the previous completed prefix, or it will be counted twice. Independently test this exact negative possibility.

The P02 algorithm still uses native `bytes.find`/ `bytes.count` for each pattern inside the block, so it does **not** eliminate Q-dependent CPU work, construct an Aho–Corasick automaton, use explicit SIMD, establish FM/rank/EF correctness, prove any cross-document `PROVEN_EMPTY`, or authenticate the source pin's author/currentness.

## Measured enwik8: exact published source

Corpus: enwik8 **100,000,000 bytes**; SHA-256 `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`; 3 balanced same-runner repetitions **per profile**. Measured medians in seconds. Note: these are narrow standalone source-scan timings, *not* GLYPH query end-to-end timing.

| Patterns Q | P01 Q separate passes, s | P02 one shared pass, s | Speedup | Source-read reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.162923 | 0.162751 | 1.001× | 0.0% |
| 2 | 0.253555 | 0.193035 | 1.314× | 50.0% |
| 4 | 0.439299 | 0.253763 | 1.731× | 75.0% |
| 8 | 0.809160 | 0.371282 | 2.179× | 87.5% |
| 16 | 1.547387 | 0.592068 | **2.614×** | 93.75% |

**Safety:** 2,119 separate query/byte-oracle comparisons over deterministic small and crossing-source fixtures; 16/16 named hostile inputs rejected; complete counts/offsets/excerpts/status matched the prior one-pattern checker on canonical enwik8 and the small oracle tests. GitHub CI job `p02-mixed-length-and-hostiles` PASSED. A freshly introduced P02 Mathlib proof has five lemmas (coverage, duplicate exclusion, exact carry coordinate) but independently checked execution must be sourced from a subsequent CI run.

**Scope of physical accounting:** source I/O only. Does not include indexed RLB2/RLR2/EF bytes, full snapshot/manifest, backup replicas, all historical versions, index-startup, construction RAM/disk, OS cache effects, trusted currentness, or witness signatures. Do not infer a 2.614× improvement to full GLYPH Vault, all searches or persistent index runtime. The 30% authenticated-storage target remains separate.

## Discovered reusable principle

**Shared verification amortization:** When multiple queries target *one already pinned immutable source*, the full bytes can be authenticated once for all queries, rather than Q times. It is unsafe to cache or share `PROVEN_EMPTY` across different snapshot roots, document revisions or incomplete leaf sets. The new algorithm's actual invariant is `one_source_read + one_complete_source_digest + all_query_states`, not `Q` independent trusted answers without any witness.

**CPU next bottleneck:** after eliminating repeated I/O, 16 queries still took 0.592 s and showed diminishing speed gains relative to Q=1. Compare (a) current native-byte-primitives-on-chunks vs (b) bounded-memory C++ Aho–Corasick / sparse DFA, possibly SIMD multi-pattern block routing. Use independent output oracle, identical chunk/snapshot scope, reproducible cold/warm p50/p95, RSS/temp disk and full-package storage costs. An AC state machine gives linear work in source bytes plus matches, but whether it wins depends on query distributions and native implementation. Preregister a falsifier (CPU slowdown, malformed bytes, coverage mismatch, memory cap breach); do not assume it wins.

**Research sources:** Daniel Lemire, [Make Your Code Faster with Data Parallelism](https://tower-research.com/exploring-data-parallelism-with-professor-daniel-lemire-at-tower/); formal arithmetic [Mathlib4](https://github.com/leanprover-community/mathlib4); run-indexed alternatives [r-index](https://github.com/nicolaprezza/r-index). These are sources of hypotheses, not evidence of GLYPH correctness.

## Governance

No new Issue, no separate PR, no merges, changes to main or frozen V1, no OVH, no operator/Composition semantics change. This P02 candidate belongs to the research folder only. Negative results and benchmarks remain visible alongside successful ones.