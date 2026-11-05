# GLYPH CORE — External Comparable Baselines and Oracle Policy V1

Recorded: 2026-11-05.
Status: RESEARCH BENCHMARK PLAN, NOT A PERFORMANCE CLAIM.
Reference design: \`real_operator_comparable_gate.py\`, real Composition Reference V1 pinned to \`81628f843dc8aa633c65594a66d2386bfc3cb5c2\`.

## Three different "references" are needed

1. **Correctness truth oracle:** raw bytes of every independently identified document; exhaustive per-document byte-window scan, including overlap, NUL, empty/duplicate documents, cross-boundary negatives and canonical global \`(doc_id, doc_offset)\`. An implementation with the same FM-index algorithm does not count as an independent byte oracle.
2. **External bitmap implementation:** CRoaring via pinned \`pyroaring==1.2.0\`. Verify intersections of document-ID candidate sets with immutable field masks; account for a separate signed/pinned field metadata commitment in any future *authenticated* comparison. Roaring's fast bitmap intersection by itself proves neither source-byte identity nor freshness nor permission security.
3. **Compressed exact-search implementations:** independent r-index and SDSL-lite, each compiled from an exact pinned upstream revision, with the same corpus and query workload; do not claim comparative speed/size without actually running them.

## External implementations selected

| Candidate | External source | Compatible task | Required adapter / limitation |
|---|---|---|---|
| r-index (Prezza) | https://github.com/nicolaprezza/r-index | Exact count and locate over repetitive texts | Upstream \`ri-count\`/\`ri-locate\` benchmarking CLIs discard per-hit coordinates; must add a standalone evidence-printing adapter or use a verified API. No oracle parity can be inferred from a zero-exit CLI alone. |
| SDSL-lite | https://github.com/simongog/sdsl-lite | \`csa_wt\`, count, locate, extract; benchmark/reference index | Implement minimally intrusive external wrapper to emit all coordinates with source/query hashes, count all index files; define terminator handling for embedded NUL. |
| CRoaring | https://github.com/RoaringBitmap/CRoaring and https://pypi.org/project/pyroaring/ | Exact field-mask membership/intersection and serialization | Use known document IDs and a cryptographically bound, immutable field metadata vector. Compare filtered result with raw-byte oracle, not only with another bitmap. |
| Raw baseline | Python byte-window implementation (and optional independent native \`memmem\` for performance) | Exact byte oracle, exhaustive including overlapping matches | No compressed index; no separate metadata trust or proof semantics. Only a correctness and operational scan control. |

Pizza & Chili's repetitive corpus, including versioned documents/source and DNA, is an established scientific text-index testbed; enwik8 and Silesia are useful but a single continuous corpus cannot demonstrate document-level filtering/duplicate/empty version semantics. Preserve original raw files; construct a *specified ordered document manifest* and count any necessary terminators/sidecars.

Upstream sources:
- https://pizzachili.dcc.uchile.cl/repcorpus.html
- https://github.com/simongog/sdsl-lite/blob/master/benchmark/README.md
- https://github.com/nicolaprezza/r-index
- https://pypi.org/project/pyroaring/

## Fair comparison protocol

- Pin source URL, exact external commit/tag, compiler flags, CPU model, SIMD availability, runtime environment and query workload hash. Do not take GitHub "latest" as a reproducibility parameter.
- Two corpus profiles are necessary:
  **A.** one physically continuous source for basic count/locate/size comparisons; 
  **B.** an ordered multi-document corpus with verified boundaries, metadata and partial-content rules for complete GLYPH semantics.
  A single-string external index with a sentinel cannot silently answer Profile B by physical concatenation; either index each document or add a documented equivalent boundary treatment and account its space/cost.
- Require completely enumerated exact hit positions, not just total counts or last log line. Compare with independent byte oracle and source hashes. Include intentionally absent query and positive query in a document excluded by the field mask; absence in filtered public view must not imply absence in the entire corpus.
- Measure **full retained system bytes** for all search/index, locate, compressed payload needed for recovery, manifests and proof, version/field data and runtime packaging. Provide kernel-only numbers separately, never add unrelated 0.336× V0 and 1.088× OVH values.
- Collect build peak RSS/temp disk/build time, open RSS/startup, count/locate/extract p50/p95/p99 from pre-registered samples, bytes decoded and independent verifier overhead. Cold, warm and cross-host are separate profiles.
- A non-proof system (r-index/SDSL) must not be penalized by pretending it provides a GLYPH portable evidence bundle. Explicitly compare base index capabilities, then **full equivalent authenticated retrieval feature sets** using documented additional proof components or mark "not provided".
- Any FALSE EMPTY, missing block accepted, mismatched coordinate, wrong SHA, incomplete coverage or altered root pin accepted is an automatic FAIL. Do not substitute average accuracy or semantic recall metrics for exact-search proof.
- Status words: \`SOURCE_ORACLE_PASS\`, \`INDEX_PARITY_PASS\`, \`EXTERNAL_BITMAP_PARITY_PASS\`, \`FIELD_PROOF_INTEGRATED_PASS\`, \`FULL_PORTABLE_PROOF_PASS\`, \`PERFORMANCE_MEASURED\`. Passing the first three does NOT imply the latter three.

## The first executed comparative gate

\`real_operator_comparable_gate.py\` checks out Composition Reference \`81628f843dc8aa633c65594a66d2386bfc3cb5c2\` (independent historical source), builds REAL operator SA/BWT/FM blocks for the reference's nine-document fixture under both three-block partitions, verifies canonical root/source identities and every full result with the existing independent Composition Reference verifier, applies an additive immutable field mask, checks actual Roaring intersection against raw byte-window oracle, and rejects missing block, tampered root, tampered mask, stale external pin and a fake \`PROVEN_EMPTY\` even when the receipt is rehashed. This is an initial low-scale **semantic comparison**, not a speed result.

The external pin is supplied out of band to the separate verifier subprocess by the test driver; it is **not** a trusted remote witness, signed currentness authority, rollback-proof production pin, nor an O1–O6 field bundle. Immutable masks are research sidecars; no change is made to existing operator manifests or frozen V1.

## Next measurable gate

Once real-block correctness passes, benchmark LOC1→LOC2 on canonical enwik8 (100,000,000 bytes) independently against direct SA/raw corpus oracle, with startup/RSS and all required hostile gates. Separately compile a pinned SDSL CSA comparator to show count/locate/size cost on the same enwik8 bytes; only then assess whether a native SIMD rank implementation meaningfully moves the Pareto frontier. Never run a 1 TB or GPU benchmark as a substitute for unresolved proof identity/coverage.
