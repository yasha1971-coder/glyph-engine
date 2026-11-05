# GLYPH CORE CONVERGENCE — G0 / Searchable-Compression Frontier

Date: 2026-11-05  
Status: PRE-REGISTERED RESEARCH PROTOCOL; NO NEW ENGINE RESULT OR WINNER CLAIMED.  
Base: glyph-engine PR #15 at 425184816605818821b7043851fad5e68c01d8c5.  
Authority: research proposal. Public V1 semantics, ABI freeze, frozen V1 artifacts and Personal Memory gates stay unchanged.

## Decision: recover the global objective

GLYPH CORE's research target is *verifiable searchable compression*: losslessly preserve committed bytes, perform exact count/locate without a whole-archive-decompression requirement, independently verify selected results, and restore exact bytes. Measure the joint cost of the entire searchable artifact, including index, authenticated evidence, runtime resources and recovery state.

Personal Memory is a downstream application, NOT the original definition of GLYPH CORE.

Do not optimize compression ratio, query speed or feature count in isolation. RLBWT, Move/r-index, run-aware data structures, PAQ/cmix-style entropy coding and class-specific hybrid routes are research candidates, not completed claims. The winner may be a different physical representation for different corpus classes if logical identities and exactness remain invariant.

## CORE-G1: the first deciding experiment

Compare candidates on the same identified source documents and identical precommitted query bytes:

1. Raw-byte oracle: independent exact substring count and canonical locate, complete document coverage. A compressed archive plus decode-and-scan can serve as a NON-INDEXED negative control, not a direct-search baseline.
2. Existing RLB3X/LOC2 path: first reproduce the verified compact kernel, then independently close the *full* searchable archive accounting. The observed 2,422,268 / 7,230,581 ratio (0.3350032314) is kernel-only, never full authenticated storage.
3. Existing hybrid preservation plus searchable spine: tie corpus/generation, selection, manifest, locate results and byte materialization cryptographically. Count all retained bytes once, even if payload and search structure are separate.
4. Reproducible external baselines: seekable zstd WITH a genuine exact query path, and one pinned r-index/sr-index/Move-class system only if it supports the same query semantics. A random-read zstd API alone is not a substring index.

Public datasets, no private-data access needed: 30-object corpus (7,230,581 bytes), enwik8 (100,000,000 bytes) and all Silesia 12 files (211,938,580 bytes). Pin URL, commit, document order, full bytes and SHA-256. Preserve empty and duplicate documents and do not treat physical concatenation as a document-level oracle. Do not silently extrapolate any result to 1 TB, the VIKA corpus or a whole Personal Memory deployment.

Pre-register query IDs, byte-pattern hashes, distribution and expected coverage before testing candidates. Include present rare/frequent, overlapping, absent, source-boundary negatives, across-block positives where logically valid, and embedded zero byte cases when supported. Full count and all queried locate coordinates must match an independent byte-level oracle. An absent match under incomplete coverage is UNKNOWN, not PROVEN_EMPTY. Exact restore must match SHA-256 of the canonical ordered source.

## Comparable cost and Pareto rules

Disjoint persistent bytes, with shared storage counted once:

    full_store_bytes = payload + search_index + locate
                     + manifests_evidence + recovery_metadata
                     + other_mandatory
    full_store_ratio = full_store_bytes / raw_source_bytes

Measure build time and peak RSS, build workspace, startup cost and peak RSS, count/locate/extract raw latency samples (>=20; empirical p50/p95), bytes decoded per bytes requested, corruption/replay verdicts and full restore. Explicitly distinguish warm cache, fresh process and cold storage. Record exact source revision, environment and workload hashes. If two records differ in any of these, do not declare a performance winner.

Pareto axes initially: full persistent bytes, locate p95, open peak RSS and decoding amplification. Nondominated does NOT mean scientifically superior; preserve sample sizes, uncertainty, other axes and differences in semantics.

Known bounds from prior work:
- RLB3X compact ratio 0.335 is a single-corpus kernel-only result.
- Binary-safe RLBWT enwik8 0.987100010 is a differently scoped measured representation.
- Frozen Laptop V1 proves a bounded authenticated discovery/selection/restore path.
- Composition Reference is research-tier; not automatic product integration.
- PR #15 provides a Personal Memory pilot. Six packed-backend SKIPs and bounded-RLB3X streaming patch lineage remain open.

None of these four results can be added together as though they described one complete runtime.

## Execution order and falsifiers

| Gate | Goal | Advance condition | Stop condition |
|---|---|---|---|
| CORE-G0 (this PR) | Preserve global goal and truthful metric protocol | Small fail-closed scorecard and tests | No performance winner claimed |
| CORE-G1 | Public 30-object full-store closure and independent count/locate/restore oracle | Exact data and complete accounted numerator, memory and latency | Missing mandatory storage, false match, fabricated coverage, wrong restore |
| CORE-G2 | Full enwik8 and Silesia comparison to genuine search baselines | Identical source/workload/environment and replay evidence | Cherry-picked corpus, silent scope changes, missing control |
| CORE-G3 | Remove identified bottleneck: RLB3X build/open RSS and startup; recover historical patch identity | Byte-compatible measured improvement with negative cases | Rewritten frozen code or unsupported "bounded" claim |
| CORE-G4 | Unified authenticated search, compact preservation and byte-perfect materialization with failure injection | Independent replay, no lost acknowledged versions, no false PROVEN_EMPTY | Hidden whole-archive decode, wrong identity, corrupt published state |
| CORE-G5 | External comparison, claim/release assessment | Measured competitive frontier, evidence checked independently | If worse, change candidate/routing/hypothesis, not truth rules |

Freeze accepted releases and artifacts, NOT the research mission. PR #17 (I0 R2) and its merge decision are independent of this research branch.

## Executable structural scorecard

scorecard.py consumes JSON candidate records. It rejects incomplete full-store accounting, incompatible corpus/workload/environment, reported mismatches, missing negative cases and unsupported kernel-only comparisons. It computes ratios and Pareto dominance from reported values. Raw samples are mandatory and p50/p95 are computed by nearest rank, not accepted as typed headlines.

Crucial limitation: **it does not open the archive, run the original corpus oracle, independently hash source bytes, replay a real experiment or authenticate the supplied receipt.** Even a structurally consistent record may be fabricated. Its output label is REPORTED_FIELDS_CONSISTENT_NOT_REPLAY_VERIFIED, not VERIFIED. Separate build/search/replay artifacts and independent validation are mandatory for CORE-G1.

Run synthetic-only contract checks:

    python3 -m unittest discover -s experiments/core_convergence_v1 -p 'test_*.py' -v

When actual measured records exist:

    python3 experiments/core_convergence_v1/scorecard.py candidateA.json candidateB.json

No synthetic numbers are committed as a real benchmark or promotional result. This PR changes no current engine implementation, no existing VERIFY script and no frozen evidence.

## Literature to compare honestly

- *Fast and Small Subsampled R-indexes*, ACM Transactions on Algorithms (2025), DOI 10.1145/3750729.
- Nishimoto & Tabei, *Dynamic Grammar-Compressed Self-Index in delta-Optimal Space*, ESA 2026, DOI 10.4230/LIPIcs.ESA.2026.6.
- Zstandard seekable format: https://github.com/facebook/zstd/tree/dev/contrib/seekable_format. Compressed range-read access is NOT a generic exact substring index.

Research success means identifying and reproducing a useful combined frontier, or disproving a preferred approach and changing course while retaining correctness and all historical evidence.
