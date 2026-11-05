# GLYPH CORE — Proof-Carrying Searchable Compression Convergence

Date: 2026-11-05
Status: ARCHITECTURAL CORRECTION / RESEARCH INTEGRATION CONTRACT (NOT A RELEASE CLAIM)
Source scope: glyph-engine PR #15 base 425184816605818821b7043851fad5e68c01d8c5; G0 research draft PR #18.
Protection: no change to existing P/R/O/I0 contracts, frozen V1, historical proof/evidence artifacts or protected main.

## Invariant: verification is not a post-processing service

The primary GLYPH concept is not "small payload + fast index + an optional proof at the end."

It is **proof-carrying exact retrieval from a compressed, restorable corpus state**.
A candidate architecture is acceptable only when the byte-level result, corpus/index/generation identity, complete query scope, selected materialization and independently reproducible verification are coherent under **one explicitly versioned contract**.

Compression and physical search structures may be separate artifacts, but neither is allowed to define its own contradictory corpus identity or to report trusted success without the required evidence. The operational product must not present an unverified hit/empty result as verified. Compression is therefore a **representation choice subordinate to proven retrieval semantics and proof closure**.

A candidate's competitive achievement is not just a ratio. It is a jointly measured **(bytes, latency, memory, verified-result coverage, replay portability, recovery correctness)** point.

## The existing executable proof architecture — do not rebuild it

| Layer | Existing artifact / obligations | Verified scope / boundary |
|---|---|---|
| P1–P12 | \`docs/specs/GLYPH_PROOF_GRAPH_V1.md\`, \`tools/run_glyph_proof_graph_v1.sh\` | Executable semantic obligations for suffix/SA/BWT/FM, corpus identity, exact count/locate, document boundaries, deterministic replay, bundle closure. Scoped test/executable proof, not a universal mathematical proof of all implementation states. |
| R0–R6 | \`docs/architecture/ENGINE_OVERVIEW.md\` | Compiled binary-safe runtime and multi-document conformance; bundled replay and source-byte checks. |
| O1–O6 | \`docs/specs/GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1.md\` | Portable self-contained evidence, validated source/index/query identities, independent replay outside source repo. Compatible included Linux binaries are part of the portability boundary. |
| I0 | \`docs/specs/GLYPH_C_ABI_V1.md\`, \`verify.sh\` | Pre-freeze ABI contract checks; NOT a completed shared-library implementation. Source scan count 38→39 requires reconciled result (PR #17), not a checker bypass. |
| Composition Reference V1 | \`docs/specs/GLYPH_COMPOSITION_FOUNDATIONS_V1.md\` | Research-tier exact partition invariance and independently replayed all-block coverage; not Index Forest production. |
| Frozen Laptop V1 | \`deploy/glyph-v1/\` and profile receipts | Scoped authenticated discovery → explicit human selection → authenticated materialization → verified bytes. Not arbitrary 1 TB corpus acceptance. |

**Important:** the proof chain and actual engine implementation already exist in their declared scope. Novel compression candidates must **reuse/bridge** these boundaries, not replace them with fresh self-reported \`verified: true\` claims or a less strict checksum-only test.

## The stronger cryptographic commitment research branch is separate

Earlier docs propose a \`Corpus Root + SA Root + Build Spec Hash\` hash commitment, Merkle inclusion proofs, count/interval/absence claims, independent audit and engine-independent verification:
- \`docs/specs/GLYPH_COMMITMENT_ARCHITECTURE_V1.md\` (DRAFT).
- \`docs/specs/GLYPH_PROOF_OF_PRESENCE_POC_V1.md\` (research prototype).
- \`docs/specs/GLYPH_COMMITMENT_GAP_ANALYSIS_V1.md\` (prior-art separation).

The POC's absence witness is **not authenticated to global suffix-order adjacency**, so general cryptographic non-membership has NOT been proved. The commitment draft does not become a completed proof system simply because executable P/R/O verification is green. Neither legal truth nor source authorship nor semantic meaning is proved by exact bytes.

This boundary must be retained explicitly. Future novelty claims about a cryptographic proof construction require independent cryptographic review and prior-art analysis. The historical documents themselves do not claim to invent authenticated pattern matching, Merkle commitments or verification without running the original engine.

## ONE contract — conceptual normalized evidence envelope (not a current wire schema)

    committed ordered bytes + content/generation identity
           |
           +-> compressed payload + binding / recoverability witnesses
           |                                    |
           +-> searchable index + binding       |
           |          |                         |
           |      count + locate                |
           |          |                         |
           +-> canonical (doc_id, doc_offset)   |
                      |                         |
           authenticated candidate/selection   |
                      |                         |
           materialize / restore exact bytes <--+
                      |
           independent evidence and replay checks
                      |
               VERIFIED result or FAIL/UNKNOWN

A future normalized envelope should bind:
- \`composed_corpus_id\` (ordered logical documents, independent of partitioning);
- \`composition_root_id\` (physical layout/block commitments);
- exact query identity and canonical results;
- search-index identity, compression/profile/decoder identity and required storage artifacts;
- selection/generation pin, restored byte identity and complete coverage;
- evidence manifest and independent replay verdict.

These fields are conceptual integration obligations; do **not** silently change existing versioned hash preimages or wire formats to match this list.

A result with missing/corrupt required proof artifacts is **UNKNOWN/FAIL CLOSED**, never a verified no-hit answer. Repartitioning may change physical root while preserving logical corpus identity and matching results.

## Correct G1 experiment: demonstrate an integrated verified result

For the public 30-object corpus, independently run the same committed query workload through:
1. source bytes and external oracle;
2. RLB3X compact searchable structure (existing known kernel);
3. chosen lossless payload/materialization representation;
4. *existing* identity/evidence/replay contracts or an explicitly versioned, independently checked adapter bridging them;
5. a fresh-process verifier that does not trust the search result flags and independently re-checks exact source bytes and completeness.

Require all of these to pass together for the declared **proof-carrying searchable storage** result. A self-reported scorecard record is only a consistency diagnostic, even if it includes a stated evidence hash.

Measure complete retained storage (payload, index, locate, verification/evidence/replay, recovery metadata), startup/build RSS and latency. Missing mandatory verifier material is a failed full-store closure, not an excuse to exclude proof overhead.

Falsify with:
- mutated payload/index/source/document order/query;
- dropped and duplicated blocks / missing manifest entries;
- stale index or wrong \`composed_corpus_id\` / \`composition_root_id\`;
- altered canonical coordinates, wrong count and truncated query;
- missing or tampered evidence and replay artifacts;
- wrong generation or wrong restored bytes;
- absent-query assertion under partial coverage.

A negative test PASS means the invalid claim was rejected; it is not a proof of a general security theorem.

## Success criterion and freeze

**Joint gate PASS** means a named, independently replayed, complete exact-byte result AND its compressed representation are verified together on a pinned corpus with known limitations and full cost accounting. It does NOT automatically establish universal compression, cryptographic non-membership, formal program verification or an unprecedented design.

Freeze accepted, versioned evidence/implementation checkpoints only; preserve historical evidence (including failed trials). Do not freeze the overarching research mission. Personal Memory remains an application that must respect these same authority boundaries.

Originality is a research hypothesis: there are relevant authenticated pattern-matching, self-index and verifiable-storage systems. The potentially distinctive contribution is the **specific combined, portable, proof-carrying, compressed exact-retrieval contract with independently executable evidence**. Its first-of-kind status must not be asserted without a defined feature matrix and independent prior-art review.
