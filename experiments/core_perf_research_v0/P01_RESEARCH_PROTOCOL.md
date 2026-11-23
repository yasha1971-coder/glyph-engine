# GLYPH performance research / CORE-PERF-P01: algorithm before threads

Date: 2026-11-06. **EXPERIMENTAL, initial predictions recorded BEFORE runner evidence.**
Existing tracked work: [CORE-L10 Issue #28](https://github.com/yasha1971-coder/glyph-engine/issues/28), [Draft PR #29](https://github.com/yasha1971-coder/glyph-engine/pull/29). No additional Issue or PR, no merge, no frozen/main/OVH/CONTROL mutation. Not a public performance claim.

## Permanent engineering learning protocol

1. **Global product invariant first:** multi-year appendable Vault; immutable document/revision IDs; exact byte search and independent recoverability, complete version-scoped query coverage, reject evidence failures as \`UNTRUSTED\`. A faster answer with false \`PROVEN_EMPTY\` is a regression, never a win.
2. **Measure the real work**, not only a hot function. Record total physical retained bytes, temporary disk, cold/warm wall+CPU, all-process RSS, exact SHA identity, fragment offsets, count complete, and proof/replay cost. A two-pass per-query verifier may dominate a 1ms FM-count operation. Do not silently relabel a microbenchmark as end-to-end Vault performance.
3. **Remove redundant work before parallelizing**: avoid repeated per-query source reconstructions and needless hashing, then batch queries to amortize full-source read; only then choose SIMD, threads or Move-based LF. Force exact before/after byte oracle.
4. **Formal statements are scoped**: Mathlib kernel theorem \`ChunkCrossingProof.lean\` only proves carry boundary arithmetic. An \`openai/math\` manuscript or arXiv preprint is research input, never automatic evidence that the current C++ or Python implementation is correct.
5. **Prediction -> falsifier -> executable controlled comparisons -> receipt -> decision.** Preserve failed predictions, fail-open negatives, regression tests and unsuccessful timings. Do not optimize away verification or silently change the original verifier.
6. **Admission/router**: physical corpus class (r/n, number of immutable leaves, pattern lengths, query frequency, memory/time budget, trust scope) selects a verified profile. "Fastest" always names the corpus, system, measurements and proof overhead.

## External research, with exact scope

- Daniel Lemire, [Tower Research: Make Your Code Faster with Data Parallelism](https://tower-research.com/exploring-data-parallelism-with-professor-daniel-lemire-at-tower/) (2026-11-05): structural/data parallelism, SIMD and rethink algorithms before microtuning.
- Brown & Langmead, [arXiv:2602.11029](https://arxiv.org/abs/2602.11029) (2026): bounded average move structures, O(r)-space LF inversion, lower space on certain repetitive genomes; **not** a tested GLYPH gain.
- Gagie et al. [r-index](https://github.com/nicolaprezza/r-index): run-count-bounded exact indexes, distinct objective from a root-authenticated appendable Vault.
- Claude et al. [arXiv:1604.08897](https://arxiv.org/abs/1604.08897): different compression/time tradeoffs on repetitive document collections.
- [Mathlib4](https://github.com/leanprover-community/mathlib4): formal checker; pinned Lean 4.25.0 / mathlib commit \`1ccd71f89cbbd82ae7d097723ce1722ca7b01c33\` matches the earlier GLYPH proof gate.
- [OpenAI math](https://github.com/openai/math): manuscripts and proof artifacts have different verification stages; partial formalization is not complete proof, much less a proof of this implementation.

## Observed PRE-EXPERIMENT starting point

- L9-S standalone verifier (unchanged file \`experiments/core_l9_scale_v0/verifier.py\`) executes **two full source reads and SHA updates per query**. For canonical 100,000,000B enwik8, **200,000,000 source bytes read/query**, 1.4GB for seven benchmark queries, ~1.23–2.09s standalone scan+structural verify reported in historical L9-S CI (not exactly this P01 microbenchmark).
- L9-S complete-source scan is independent of the producer but **not** an independently authenticated FM/rank/EF proof. Its native producer can use ~372,592 KiB RSS during full source materialization; moving bytes between processors does not erase this limit.
- CORE-L6 showed 4-thread dense LF 17.642656s / 569,568kB versus streamed run-LF ~26.596536s / 372,612kB on enwik8. More parallelism/lower RAM was NOT a universal speed win; optimize based on the measured corpus class.

## P-PERF-01: one authenticated pass, plus special frequent-byte count

**Candidate is a new isolated research module** \`one_pass_scan.py\`, NOT a change to the existing L9 verifier. For each query:

- input external SHA256 & byte-length pin; refuse nonregular or symlinked files;
- read the entire source **once**, compute SHA-256 and exact byte count/offsets/excerpt SHA simultaneously;
- with pattern length m, carry only m-1 preceding bytes across source chunks; allow overlapping hits;
- a one-byte pattern uses the native \`bytes.count\` primitive for count and enumerates only the first 256 output positions; don't do millions of Python \`find\` iterations for exact count;
- **never emit a FOUND or PROVEN_EMPTY receipt before EOF+full SHA comparison+stable file signature**. If full count>256, preserve count but emit \`UNTRUSTED/LIMITED_COMPLETENESS\` for the incomplete coordinate list.
- No change in retained RLB2/RLR2/EF, source content, root, original search semantics or index read/binding contracts. The candidate standalone scanner makes **no authoritative root/currentness or independently verified index semantics claim**.

### Preregistered predictions and falsifiers

- **P01-A structural:** exact source read volume for an authenticated scan drops from \`2*n\` to \`n\` bytes, i.e. 50% less. **FAIL** if either complete corpus hash, EOF or read accounting differs. Does not imply 50% faster execution.
- **P01-B count optimization:** on canonical enwik8 one-byte \`e\` (previous count 8,001,205), the median candidate **scan wall time ≤ half** the unchanged original two-pass scan, across three same-runner paired repetitions. **FAIL** if median speedup <2x. Record actual speedup even on FAIL; no fabricated CPU SIMD assertion (native Python bytes primitives may be optimized internally, their ISA is not pinned).
- **P01-C heterogeneous queries:** candidate scan median should not exceed **1.2 × the two-pass baseline** on the 2-hit and zero-hit enwik8 patterns. **FAIL** if slower. This is an empirical prediction, not correctness gate.
- **P01-D fail closed:** every tiny binary exact byte oracle, chunk boundary, overlap, empty query rejection, missing/truncated/mutated source, bad external pin and >256-hit limited response must match independent oracle or fail. **FAIL** if any incorrect FOUND/PROVEN_EMPTY or mismatched offsets is accepted.
- **P01-E unchanged predecessor:** original L9-S verifier & frozen V1 unchanged, no change to physical indexed storage. **FAIL** on any changed original file.

### Formal guarantee scope

The \`ChunkCrossingProof.lean\` proof encodes three natural-number arithmetic obligations:
- A byte pattern of length m crossing chunk boundary b cannot start earlier than b-(m-1).
- A match discovered in this carry cannot already have been fully completed before b.
- A crossing match fully received in the next block fits inside carry + current chunk.

These lemmas are **not** a formal proof of SHA-256, Python, source mutation behavior or OS I/O and must never be labelled as one.

## Scientific successor gates, not new Issues

- **P02 batched queries:** Q independent patterns against the same pinned snapshot should be counted in a **single authenticated streaming pass**, using carefully bounded Aho–Corasick / finite state search. Aims to replace Q*n independent source read bytes with n, plus total matches and bounded result windows. Prove mixed-length carry/match semantics, preserve complete full-count vs capped coordinates, and test worst-case automaton memory with admission. Do NOT claim O(1) index proof.
- **P03 profile selection:** compare same-runner C++ dense LF, run-based, Orbit/RunPerm, SIMD/block search on actual repetitive/nonrepetitive corpora; compute Pareto frontier (total indexed package, wall, transient RAM, temp disk, exact bytes, citation, license). No untested 30% claim.
- **P04 composition/witness:** batch signed-snapshot multi-leaf full coverage, version-scoped ROOT/FOUND/PROVEN_EMPTY; a faster incomplete query is invalid. Prove authorized root/currentness, not just a parent pointer. Fit into the existing CORE-L10 task.

**Important:** P01 executable CI results to be appended here only after exact matching CI concludes; current code + this document are research hypotheses, not production upgrades.
