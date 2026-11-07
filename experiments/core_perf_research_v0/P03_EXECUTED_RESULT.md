# CORE-PERF-P03 — executed native Aho–Corasick evidence

**Date:** 2026-11-07. **Status:** `VERIFIED_P03_NATIVE_EXACT_SCAN_RESEARCH`, **not** a GLYPH Vault upgrade, not a cryptographically authenticated FM-index, not a release. Existing [Issue #28](https://github.com/yasha1971-coder/glyph-engine/issues/28) and [Draft PR #29](https://github.com/yasha1971-coder/glyph-engine/pull/29) only.

## Exact reproducer and comparator

- Code: `ac_native_v0.cpp` native C++17 using OpenSSL EVP SHA256; `p03_gate.py` independently checks results against an unchanged Python P02 byte-scan oracle and direct byte occurrences; `core-perf-p03.yml` compiles with `g++ -std=c++17 -O3 -Wall -Wextra -Werror -pedantic -lcrypto` on GitHub ubuntu-latest.
- Canonical source enwik8: **100,000,000 bytes**, SHA-256 `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`. Both candidates read and SHA the complete source once per multi-query batch. Three balanced same-runner iterations each for Q=1,2,4,8,16. Correctness checks bind pattern hex/count/full or capped offsets/truncated/FOUND/PROVEN_EMPTY/UNTRUSTED.
- Comparator is **P02 Python batch**, NOT the older two-pass L9 verifier and NOT a native FM-index.
- Original P01 source Git blob `56245f51a2dd0c4b17fb018d277b5222096448f9`; original L9 verifier Git blob `e0a4c660daa5185dd218e840a5356fceb73c61e5`. Neither was modified.
- Successful specialized [GitHub CI run **42358217285**](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42358217285), both **native-safety** and **native-100mb-paired** jobs SUCCESS at code commit `98eb971f6b750a788639556d09fe1090138e13bc` (before cosmetic CI marker fix and this report).

## Measured wall medians, seconds

| Number of patterns Q | P02 Python batch | P03 native AC | P02/P03 | Faster |
|---:|---:|---:|---:|---|
| 1 | 0.197037 | 0.468636 | 0.420× | P02 |
| 2 | 0.231383 | 0.469966 | 0.492× | P02 |
| 4 | 0.273924 | 0.468666 | 0.584× | P02 |
| 8 | 0.392871 | 0.468089 | 0.839× | P02 |
| 16 | 0.642034 | 0.470225 | **1.365×** | P03 |

**Preregistered hypothesis P03-C** was native faster for Q=16: **PASS** (1.365×), but P03 was **2.38× slower for Q=1**, and slower at Q=2,4,8. Never promote AC to universal default based on this dataset. Cross-over observed **between Q=8 and Q=16 for this single pinned corpus on this runner**; not a validated hard-coded router threshold. The absolute wall measurements vary across GitHub runners; cross-test comparisons must stay within paired runs. This is source scan/authentication only and omits physical indexed RLB2/RLR2/EF startup, source reconstruction, proof verification, snapshot metadata, disk materialization and trusted root authority.

## Exactness and resource admission

- **2,830 exact oracle comparisons** spanning biased repetitive/high-entropy small binary inputs, 5 MiB repetitive stress, query overlap, NUL, 0xFF, and positions crossing 1MiB input chunks.
- **16/16 negative tests rejected** with no authoritative answer, including wrong digest, wrong source geometry, missing source, symlink/directory, changed bytes, malformed hex, duplicate/empty patterns, 17th query, oversized total pattern memory and unknown CLI option.
- Native AC accepts at most **16 nonempty patterns, 4,096 total pattern bytes, at most 4,097 states** with fixed 256-way u32 transitions/state and a 16-bit-equivalent bitmask output; the gate enforces <=5 MiB dynamic automaton allocation. It uses one 1MiB input buffer and at most 256 emitted offsets per pattern, but computes full count. The native child peak RSS and CPU measurements are in the CI `p03-enwik8/result.json` artifact; do not equate the theoretical <=5 MiB automaton budget with measured total process RSS.
- After complete EOF, pinned SHA-256 and stable source device/inode/size/mtime_ns/ctime_ns are checked. Results are only printed at the end. Changed or untrusted source produces exit 3 / UNTRUSTED. External pin is **unsigned**; these checks do not prove absence of a malicious race or external currentness.
- The AC machine carries state across physical blocks (no carry bytes), so source-offset computation uses `end_position - pattern_length`. Independent byte-oracle, rather than new Lean code, currently checks the native implementation. The pinned P02 Mathlib proof applies to P02's shared carry arithmetic only, **not** AC correctness.

## Failures and fixes preserved

1. First specialized [run **42358010887**](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42358010887): benchmark SUCCESS, negative-safety job FAILED. The test accidentally omitted the executable path, invoking the source file as a program (`PermissionError`). This was a **test-harness error**, not an observed false empty or wrong offset. Fixed CLI construction in `p03_gate.py`.
2. Second [run **42358113896**](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42358113896): benchmark SUCCESS, safety failed on a *test expected-count mismatch*: the independent memory-admission rejection was omitted from the hostile total. The tested 16 hostile cases had been rejected, but the test asserted 15. Fixed count 16 in code/CI; first red attempts retained.
3. Third [run **42358217285**](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42358217285) succeeded for both jobs. Subsequent cosmetic marker report now reads `16_HOSTILES`; use this run for the exact primary data and subsequent head CI for final SHA verification.

## Engineering decision and next measurements

**Select by measured workload, not prestige.** For one to eight queries here, native Python `bytes.find/count` after shared SHA is faster than AC. At 16 queries, one forward AC automaton wins 1.365×. Both have mandatory full source I/O; the source scan is **not** a succinct proof. A candidate router is a *research hypothesis* conditioned on Q, total pattern bytes, alphabet entropy, match frequency and available RAM, not a shipping heuristic.

Next test should use actual VIKA/Vault *nonprivate reproducible classes*, low-/high-entropy and high-overlap corpora, plus cold/warm and query batching with fixed total byte scope. Separate the costs for: source restoration, verified root/leaf coverage, native index count, independent verification, query scanning, full package, startup and memory. Prove any profile admission and fallback respects `FOUND/PROVEN_EMPTY/UNTRUSTED` before product integration.

Do **not** repeat an already proven P03 benchmark in new Issues/PRs. Keep all experiments within the current research track. Frozen V1, main, OVH, CONTROL, production, release and signed/currentness claims unchanged.
