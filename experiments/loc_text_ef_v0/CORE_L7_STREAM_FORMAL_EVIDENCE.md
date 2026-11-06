# CORE-L7 — bounded-buffer exact source materialization and formal run law

Date: 2026-11-06
Status: **EXECUTED 8MiB RESEARCH PASS; 100MB CI separate and pending final verdict**
Location: [Draft PR #21](https://github.com/yasha1971-coder/glyph-engine/pull/21); historical source/format frozen commit \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`. No source branch/main/OVH/public release modification.

## Global objective, explicitly preserved

The goal is one **versioned, compact, independently verifiable exact-search and source-reconstruction state** with \`FOUND / PROVEN_EMPTY / UNTRUSTED\` evidence semantics, human-confirmed materialization, immutable history, and documented resource limits. A new low-RSS decoder serves this *resource* axis; it is not an independent P/R/O proof implementation.

CORE-L6 showed that transient LF navigation can scale with BWT run count r, but still held a full uncompressed output vector of size n in RAM. CORE-L7 avoids that n-byte buffer by reconstructing text-sampled windows into ≤1MiB per-worker chunks and \`pwrite\`ing disjoint extents in a hidden temporary file. Only after all EF-to-EF LF boundary checks succeed does it \`fdatasync\`, close and hard-link to a new output candidate name with no overwrite. An external runner then checks exact original source SHA-256. Failed input/decode never publishes a candidate output. The candidate itself is **not a trusted/cryptographically authenticated P/R/O materialization** until an independent root-bound verifier exists.

No extra persistent source archive, restart table or second on-disk index was introduced; input is the unchanged RLB2/EF from prior gates. Original RLR2 participates in the larger search state but is unnecessary for full source inversion.

## Formal verification: mathematical scope, not software scope

New \`RunIntervalLFProof.lean\` defines abstract prefix rank and LF over a BWT sequence. For each run of constant symbol c and offset d, a Lean proof shows:

\`\`\`text
rank_c(a+d) = rank_c(a)+d
LF(a+d) = LF(a)+d
\`\`\`

[Lean 4.25.0 + pinned Mathlib 1ccd71f CI 42141468659 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141468659). The initial proof attempt failed due to rewrite orientation; the corrected Lean file was checked by the kernel. This is standard BWT mathematics, already known. It does not establish valid native parsing, cryptographic identity, sorted interval predecessor, fail-closed absence, canonical coordinates or host integrity.

Research methodology comparisons:
- [Mathlib4](https://github.com/leanprover-community/mathlib4): use a pinned proof assistant as an independently executable mathematical gate.
- [OpenAI math](https://github.com/openai/math): separate unformalized research arguments from Lean-checked claims; its published methods do not grant any automatic GLYPH proof.
- [Orbit / SEA 2026](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2026.9): existing O(n)-time/O(r)-working-space BWT inversion via bounded move intervals; the current CORE-L7 technique is not an implementation of that optimal algorithm.
- [Move-r](https://github.com/LukasNalbach/Move-r): provides count, locate and parallel full source reconstruction already; GLYPH's possible advantage has to be measured in trust/provenance and complete storage/latency frontier, not generic functional novelty.

## Small binary source — PASS

[CI 42141474105 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141474105): 18/18 exact source reconstructions (text sample sizes 2/3/5, modes rank/direct, 1/2/4 threads), 24-byte original binary-safe fixture including NUL and 0xFF. Three deliberately corrupted sources rejected: RLB2 payload bit flip, EF magic corruption and truncated EF. Initial compiler diagnostic (unused referenced helper and unused timing variable) was recorded in failed prior CI 42141337487; fixed without touching frozen source or algorithm.

## Repetitive 8MiB source — PASS, 3 repeats per profile

[CI 42141603791 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141603791): 128 generations of a 64KiB source with 32 point edits each; exact original source 8,388,608B, 24,785 BWT runs, \`r/n≈0.0029546\`. Every one of **30/30** full reconstructions matched the same source SHA-256. The external driver reconstructed the frozen RLB2/RLR2 and EF from exactly pinned builders, deleted original source, SA and BWT before every decoder call and used an SHA-based source oracle. Three counterbalanced runs per mode.

| Method | Threads | Median full wall (s) | Maximum peak process RSS (kB) |
|---|---:|---:|---:|
| Original sequential dense LF | 1 | 1.258834 | 46,008 |
| Original dense LF rank | 4 | 0.420916 | 45,596 |
| Original run LF buffered/direct | 4 | 0.388765 | 13,100 |
| **New run LF streaming/direct** | **4** | **0.420099** | **8,876** |
| New run LF streaming/rank | 4 | 0.609106 | 8,868 |
| New run LF streaming/rank | 1 | 1.270248 | 5,676 |

The streaming/direct 4T median is nearly equal to old dense 4T on this *specific* synthetic source (0.420099 versus 0.420916s), while process max RSS falls from 45,596 to 8,876 kB (~80.5% reduction, ~5.14x less). Its wall time is slightly **slower** than buffered/direct run LF (0.388765s). Neither comparison justifies a broad fastest/runtime/production claim. The full retained RLB2+RLR2+EF three-file subtotal is 1,350,543B for this fixture, **excluding** proof, code, external witnesses, replicas and user metadata.

## Memory model and boundary of theorem

Current experimental resident allocations include:

\`\`\`text
RLB2 encoded compressed payload bytes
+ 8*r bytes (run starts + LF target per run)
+ optional r bytes (direct run symbols)
+ 4*b bytes (coarse predecessor hints, b<=r)
+ entire physical EF index
+ 4*m bytes (in-memory quotient -> row map, m≈n/s)
+ <=threads * 1MiB (simultaneous write buffers)
+ allocator, std::vector and execution overhead
\`\`\`

So output reconstruction no longer needs a separate n-byte RAM vector, but **TOTAL process RAM is not rigorously O(r)** because EF sample count is O(n/s). This runtime is still limited to n<2^32 rows, so 1TB is not established. In addition, a 4T direct-symbol mode makes predecessor queries expensive when r/n≈0.366, as shown by the same-run enwik8 CORE-L6 results. A measured router cannot be replaced by an unvalidated hard-coded cutoff.

## Original canonical enwik8 100MB gate

[CORE-L7 enwik8 CI 42141692079](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141692079) launched a same-run 9-profile (original sequential, dense 1T/4T, streaming run rank/direct 1T/2T/4T) full 100MB SHA, pinned 73.365MB RLB2/12.844MB RLR2/2.844MB EF, actual peak RSS and wall. **COMPLETED: [CI 42141692079 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141692079).** All **9/9** 100,000,000-byte enwik8 full source restorations SHA256-equal to externally pinned `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`, with source/SA/BWT removed before restore. Same-run 4T results: original dense LF **17.642656 s / 569,568 kB peak RSS**; streaming run-rank **26.596536 s / 372,612 kB peak RSS**; streaming run-direct **25.910551 s / 408,364 kB peak RSS**. Single measured run per profile, no durable native production performance claim. This is a memory/speed Pareto tradeoff, NOT a simultaneous speed-and-memory win on enwik8.

## Next falsifiable gates

1. Qualify 100MB streaming on canonical enwik8, then independently repeat on a real multi-version external corpus; avoid cherry-picked synthetic alone.
2. Add source-bound SHA-256 and required-root witness to the actual *candidate publication* protocol; until then the encoder's FNV integrity check is not an authentication proof.
3. Restore memory admission: reject input before allocations if estimated resident RLB2 + run index + EF + scratch exceed the configured memory budget. Prove and test a finite arithmetic bound in Lean and C++ (two distinct scopes).
4. Compare independent pinned Orbit/Move-r source restoration and count/locate with identical input, toolchains, pinned hashes and complete physical package accounting. No GPL code incorporation without owner/licensing decision.
5. Lift u32 n/r geometry, develop 64-bit file and segment model for terabyte-scale without silently enlarging permanent on-disk metadata.
6. Close full P/R/O evidence: immutable source identity, authenticated exact query excerpts and canonical coordinates, independent portable verifier, realistic corrupted/rollback attacks, honest production packaging.

**Owner strategic veto preserved:** no release or merge, OVH deployment, frozen format redefinition, public claims, human-source ingest or C ABI implementation authorized.


## P-L7-001 — prospective 100MB output-RAM prediction (recorded before reviewing CI result)

This is a **prediction made while GitHub CI 42141692079 was still running**, not a prospective preregistration made before the CI launch.

Current buffered run-LF enwik8 rank4 same-run reference from [42139617303](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42139617303): peak **466,152 kB**. The buffering path held the entire **100,000,000-byte output** in RAM. CORE-L7 removes this allocation, adding at most four 1MiB write buffers. Since page cache and allocator interactions can confound RSS, exact savings need not equal byte-for-byte difference.

**Predict**:
- P-L7-001A: streaming run LF rank4 enwik8 produces **identical source SHA-256** and peak RSS **under 400,616 kB** (saving more than 64MiB vs 466,152kB historical run-LF reference). FAILURE if output SHA mismatches, the CI exits nonzero, or measured peak RSS is not below this threshold.
- P-L7-001B: persistent RLB2+RLR2+EF stays **89,053,457 bytes**; no extra source payload/restart file. FAILURE if any additional retained source-side artifact is needed or a counted file differs in size/hash.
- P-L7-001C: 4T wall time is **UNKNOWN**. Measure directly against original dense 4T and run-LF buffered where available; no speed claim is implied by RAM savings.
- P-L7-001D: even if PASS, neither input parsing nor output SHA checking is yet an independent cryptographic P/R/O authority; store final status as research PASS only.

The threshold 400,616 kB is 466,152 minus 65,536 (64MiB in kB). This is an empirical process-RSS prediction, not a theorem. It will be superseded by the exact same-run measured result, retaining any prediction miss.

**POST-RESULT VERDICT (2026-11-06):** [CI 42141692079 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141692079), rank 4T streaming **372,612 kB**, therefore **P-L7-001A PASS**, 28,004 kB below threshold and **93,540 kB lower** than historical buffered run4 peak (466,152 - 372,612). P-L7-001B PASS for unchanged RLB2+RLR2+EF physical files; exact 4T wall 26.596536s, so no speed victory over dense4. 100MB SHA verified 9/9, P-L7-001D retains research scope and excludes signed P/R/O attestation. Note: the root and source were **not** cryptographically verified before publication by the native CORE-L7 decoder itself; that is a subsequent CORE-L8 gate.
