# CORE-L5 — executed parallel BWT inversion, canonical enwik8 100 MB

**Date:** 2026-11-06 (Berlin local). **Status: EXECUTED EXPERIMENTAL PASS; not production.**

**Immutable historical source:** `rlbwt-binary-safe-v2` commit `c33cb493bc75f13ee4a904fabb190a29a196c94b`.
**Executed code head:** `e724ee453ebd55d334a149519aabb7417ca7cec3`, additive research Draft PR #21.
**CI:** [GitHub Actions 42128864632 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42128864632). JSON-only receipt artifact ID `13461589387`.

## Experimental method and source identity

One GitHub-hosted runner, source canonical Matt Mahoney enwik8: 100,000,000 bytes, SHA256 `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`.

Regenerated frozen SA, BWT, RLB2 and RLR2 hashes byte-for-byte. Rebuilt the research text-sampled Elias–Fano file (SHA256 `c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7`) from canonical SA, verifying all 781,251 sample records. Deleted raw source, 400 MB SA and 200 MB BWT **before running any decoder**.

**Persistent search/recovery state:** RLB2 73,365,105 B + RLR2 12,844,028 B + EF 2,844,324 B = **89,053,457 bytes (0.89053457×)**. Native parallel inverse consumes RLB2 and EF; query/range code additionally consumes RLR2. No separately stored source archive or restart map. The q→SA-row lookup is reconstructed from the EF stream in transient RAM and accounted as ~3.125 MB.

Each decoder wrote and independently SHA256-confirmed **all 100,000,000 bytes**. The parallel C++ decoder verifies LF transitions across every 128-byte text window against the adjacent EF sampled row, so sample transitions and reconstructed bytes are checked without retaining construction SA. This is not itself a cryptographic global nonmembership proof.

## Same-run measurements (each profile executed ONCE)

| Mode | Threads | Full wall (s) | LF inversion stage (s) | Peak RSS (kB) | Full source SHA256 |
|---|---:|---:|---:|---:|---|
| historical C++ sequential baseline | 1 | 29.849930 | 19.741399 | 575,928 | PASS |
| parallel EF / rank | 1 | 30.328391 | 20.023563 | 569,628 | PASS |
| parallel EF / rank | 2 | 20.379847 | 10.058896 | 569,568 | PASS |
| **parallel EF / rank** | **4** | **17.633091** | **7.255620** | **569,568** | **PASS** |
| parallel EF / direct BWT byte cache | 1 | 25.845640 | 15.365634 | 667,292 | PASS |
| parallel EF / direct BWT byte cache | 2 | 18.157838 | 7.747460 | 667,192 | PASS |
| parallel EF / direct BWT byte cache | 4 | 17.131035 | 6.664308 | 667,172 | PASS |

**Measured one-run speed ratios:** 4-thread rank is **1.693× end-to-end** and **2.721× LF inversion stage** versus same-run historical sequential baseline. 4-thread direct is **1.742× end-to-end** and **2.962× LF stage**, but adds ~95 MiB peak RSS compared with rank mode.

Observed LF preparation dominates at around **10.1 seconds** irrespective of worker thread count. Simply increasing threads cannot remove this serial phase; this is the next falsifiable bottleneck.

## What is proven and what is not

Proven for the pinned 100MB source and executable: a native decoder can exploit the already retained EF text-position samples as independently decodable 128-byte windows; the full input is byte-exact reconstructed using 1, 2 or 4 threads, with matched sample boundaries and without second source payload.

**NOT proven:** statistically robust throughput or p50/p95/p99 (single pass each), 1TB throughput, bounded transient memory (native still uses O(n) u32 LF map and O(n) output), independent signed O3/O4 evidence, currentness/fork/rollback, byte extraction in native production ABI, multi-document completeness, universal compressed-state ratio, novelty versus other compressed source-restoring indexes, or production safe host/parser hardening.

**External prior art is explicit:** [Move-r](https://github.com/LukasNalbach/Move-r) supports count/locate and multithreaded exact-text revert. Independent [Orbit](https://github.com/drnatebrown/orbit) and [SEA 2026 move structure paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2026.9) develop O(r)-working-space BWT inversion. Thus parallel inversion/searchable recovery itself is not a world-first; GLYPH's value must be demonstrated through its canonical proof-carrying contract plus a measured Pareto frontier.

## Chosen engineering next gate

**CORE-L6 LF construction/storage:** isolate serial ~10-second LF build, compare dense 4N-byte LF with run-based move / length-capped interval variants on both enwik8 (r/n ~0.3666) and much more repetitive corpora. Measure construction time, source-restore wall, peak RSS, complete retained file cost and full SHA. Do **not** append uncounted tables to the permanent index. Preserve independently pinned RLB2/RLR2/EF, P/R/O identity and fail-closed behavior.

**Strategic veto:** Draft PR #21 not merged; frozen source, main, OVH, ABI, demo and production unchanged.
