# GLYPH Preservation Register V1

Status: **DRAFT INVENTORY — FAIL CLOSED**  
Established: **2026-08-14**  
Authority: **yasha1971-coder**  
Logical identity: `glyph-object:governance:preservation-register:v1`  
Governed by: `GLYPH_PRESERVATION_LAW_V1.md`

## 1. Purpose

This register records durable GLYPH material that must not disappear merely
because it is old, unused by the current runtime, superseded, difficult to
classify, or not yet connected to the current proof chain.

It is an inventory, not a claim that every recorded item is correct, current,
independently verified, production-ready, or part of the canonical core.
Preservation and endorsement are different decisions.

This document does not create a second GLYPH core. `main` remains the only
integrated canonical state. Development branches are proposals or historical
pointers. They may contain durable knowledge without becoming parallel cores.

## 2. Governing interpretation

The following rules apply to every item recorded here:

1. `PRESERVE` means retain the durable record and its identity; it does not mean
   merge, deploy, advertise, or certify it.
2. Absence from this register means `UNKNOWN`, not safe to delete.
3. A branch with zero commits ahead of `main` may still be a valuable historical
   name, review boundary, release boundary, or provenance pointer.
4. A tag is a durable publication pointer. Its name must not be reused for a
   different commit.
5. A failed experiment and a negative result are scientific evidence when their
   scope, inputs, and result remain recoverable.
6. External sources remain valuable even when they are not cited by the current
   design. Non-use is not grounds for erasure.
7. Important movement is recorded in `GLYPH_MOVEMENT_LEDGER.md`; movement is not
   deletion and must preserve identity and discoverability.
8. Only the Owner may authorize deletion or destructive replacement under the
   Preservation Law.

## 3. Snapshot boundary

This inventory was observed from GitHub on 2026-08-14 before the first
publication of this register.

| Field | Value |
|---|---|
| Repository | `yasha1971-coder/glyph-engine` |
| Canonical branch | `main` |
| Canonical head | `9f79fd5cf8f969ed46c2c1ad7945f6b5a2944edb` |
| Inventory branch | `centenary-247-v1` |
| Inventory input head | `87e2157e1b1f2361279c99016b3412a0ead7fd81` |
| Tracked tree entries on `main` | 697 |
| Tracked blobs on `main` | 643 |
| Branches observed | 12 |
| Tags observed | 12 |
| GitHub releases observed | 2 |
| Main protection | active ruleset; deletion and non-fast-forward updates blocked |

The counts are observations at one boundary, not permanent expected constants.
Future inventories append a new dated snapshot or revision; they do not rewrite
this boundary into a different historical state.

## 4. Branch register

`Ahead` and `behind` are comparisons with `main` at the snapshot boundary.
Ahead counts are not disjoint across branches: one proposal may contain the
history of another proposal.

| Branch | Head | Relation to `main` | Preservation classification |
|---|---|---:|---|
| `main` | `9f79fd5cf8f969ed46c2c1ad7945f6b5a2944edb` | canonical | `CANONICAL_INTEGRATED`; protected |
| `centenary-247-v1` | `87e2157e1b1f2361279c99016b3412a0ead7fd81` | ahead 11 | `PROPOSAL_UNIQUE`; governance and preservation work |
| `composition-foundations-v1` | `c6c9be67033c09de8deef7003e31049b5187c10a` | ahead 15 | `PROPOSAL_UNIQUE`; foundations plus composition chain |
| `composition-reference-v1` | `c4138812f60704083f099fe0daeb6461abc9d0c2` | ahead 13 | `PROPOSAL_UNIQUE`; executable M01–M25 closure |
| `composition-semantics-v1-spec` | `5e67d0a3178581414356aa5c12e217e93d7abcea` | ahead 2 | `PROPOSAL_UNIQUE`; original semantics boundary |
| `embedded-i0-second-review-v1` | `dff6127b0758661c8cf36456dbd4e40a671122c6` | ahead 2 | `REVIEW_PACKAGE_UNIQUE`; independent-review input, not an approval |
| `embedded-api-v1` | `ff7ab6b30a84e4691dc86d61a6d65e76d4b2a0df` | behind 2 | `HISTORICAL_BOUNDARY`; content ancestry retained in `main` |
| `feature/segmented-v0.2` | `dc124b23e3f2b0e31fed654e0241fe9010aef21f` | behind 344 | `HISTORICAL_RESEARCH_BOUNDARY` |
| `operator-path-v1` | `8e8019b42a9c23e8d0b9d1c81ab815503952f877` | behind 11 | `VERIFIED_HISTORICAL_BOUNDARY`; also tagged |
| `public-surface-v1` | `192307b5a800ae3567ff036bbfc1fc5e34f091a4` | behind 1 | `PUBLICATION_BOUNDARY` |
| `runtime-conformance-v1` | `77aaf82d75d9fba848ee14a7c34426b37103da3a` | behind 18 | `VERIFIED_HISTORICAL_BOUNDARY`; also tagged |
| `site-public-surface-v1` | `9f79fd5cf8f969ed46c2c1ad7945f6b5a2944edb` | identical | `PUBLICATION_ALIAS_BOUNDARY` |

### 4.1 Unique proposal content requiring explicit integration decisions

- `composition-semantics-v1-spec` contains the first composition semantics and
  architecture boundary.
- `composition-reference-v1` contains the deterministic reference checker and
  exact M01–M25 mutation closure.
- `composition-foundations-v1` adds the mathematical foundations document over
  the composition chain.
- `embedded-i0-second-review-v1` contains
  `GLYPH_EMBEDDED_I0_SECOND_REVIEW_BRIEF_V1.md` and
  `GLYPH_EMBEDDED_I0_SECOND_REVIEW_MANIFEST_V1.json`. It is a prepared review
  package and must not be misreported as completed independent review.
- `centenary-247-v1` contains the Preservation Law, Movement Ledger, 247-facet
  draft standard, identity/dependency model, object graph, and structural gate.

No branch in this table is authorized for deletion by this register.

## 5. Tag register

| Tag | Commit | Recorded role |
|---|---|---|
| `v0.1` | `1919d9ccef29b5cfd65ab47d05597c2654153b05` | prototype release boundary |
| `v0.1-stable` | `2c803812e1d6bff29ad63a0d0d24a4495bf25f73` | named historical stability boundary |
| `v0.1-alpha` | `6426fc12fd3648288d6fd0bb01936d0cc38f0d02` | alpha release boundary |
| `structural-fingerprint-v0-verified` | `0f323c3bfa88958baec0973d51af032151553a93` | structural fingerprint verification boundary |
| `sentinel-safe-v1` | `8cca2af8f4deed16734d39e3399b2cfb14098c06` | sentinel-safety boundary |
| `rlbwt-bounded-evidence-v1` | `d9ae6b0de3e486e4c5db112c8d9621857b8a25ea` | bounded-evidence boundary |
| `retrieval-v1` | `e05070687e80d96efcaed12f5ac51421c7d9cd0d` | retrieval contract boundary |
| `glyph-proof-graph-v1-verified` | `227a4503189bc517ee7d1feed80a24149bef6c60` | P1–P12 proof-graph boundary |
| `glyph-operator-path-v1-verified` | `8e8019b42a9c23e8d0b9d1c81ab815503952f877` | O1–O6 operator boundary |
| `glyph-embedded-i0-second-review-v1` | `dff6127b0758661c8cf36456dbd4e40a671122c6` | second-review package boundary |
| `glyph-binary-runtime-v1-verified` | `77aaf82d75d9fba848ee14a7c34426b37103da3a` | R0–R6 runtime boundary |
| `evidence-bundle-v1` | `e4fbb5f810a7456d3db6331cf8c7998b3ad72203` | evidence-bundle boundary |

The words `verified`, `stable`, or `evidence` in a historical tag name record
the scope claimed at that boundary. They do not automatically prove broader
current production readiness.

## 6. Release register

| Release | Tag | Published | State at snapshot |
|---|---|---|---|
| `GLYPH v0.1 — Exact Byte Retrieval Prototype` | `v0.1` | 2026-05-05 | prerelease; mutable; no attached assets |
| `GLYPH v0.1-alpha – byte-exact retrieval prototype` | `v0.1-alpha` | 2026-05-02 | prerelease; mutable; no attached assets |

The releases are preserved as historical public claims. Their current lack of
immutability and attached assets is recorded as a supply-chain gap, not repaired
retroactively by this document.

## 7. Durable repository families

The following top-level families are preservation roots. Counts are tracked
blobs on `main` at the snapshot boundary.

| Family | Blobs | Preservation role |
|---|---:|---|
| `tools/` | 221 | builders, checkers, replay and research tooling |
| `benchmarks/` | 121 | protocols, machine scope, results, negative and positive measurements |
| `docs/` | 116 | specifications, architecture, review and publication records |
| `src/` | 36 | runtime implementation |
| `REFERENCE_BENCH/` | 32 | public reproducibility reports, queries, proofs and archived hashes |
| `research/` | 22 | exploratory work and prototypes |
| `archive/` | 21 | historical implementations and site states |
| `tests/` | 17 | executable behavioral boundaries |
| `third_party/` | 14 | vendored dependency provenance and notices |
| `examples/` | 11 | reproducible operator and evidence paths |

Also preserved are `.github/`, `manifests/`, `bench_queries/`, `config/`,
`include/`, `site/`, the root specifications, notices, entrypoints, and
`verify.sh`.

Directories named `archive`, `legacy`, `research`, or `results` are not cleanup
targets merely because of their names.

## 8. Core proof and evidence spine

The following families are high-value identities that must be promoted into the
object graph only after their exact revisions and relations are audited:

- canonical verification entrypoint: `verify.sh`;
- proof graph: `docs/specs/GLYPH_PROOF_GRAPH_V1.md`, its runner, and
  `benchmarks/results/GLYPH_PROOF_GRAPH_V1.json`;
- binary runtime conformance graph and evidence bundle;
- operator conformance graph, manifests, query, runtime index, workflow, and
  operator evidence bundle;
- embedded I0 contract plus pre-freeze and second-review packages;
- evidence object, export, replay, corpus fingerprint, evidence case, evidence
  bundle, bundle replay, and portable replay specifications and reports;
- RLBWT bounded evidence schemas, builders, verifiers, replay tools, tiny
  fixture, and reports;
- structural fingerprint V0 schemas, examples, and replay outputs;
- Composition Semantics V1, Composition Foundations V1, the M01–M25 reference
  checker, and future Composition Evidence V1;
- public benchmark protocols, machine specifications, query sets, corpus
  provenance, cold/warm state, and raw result artifacts;
- archived SHA-256 manifests and storage snapshots under
  `REFERENCE_BENCH/ARCHIVE/`.

This section is a promotion queue, not a declaration that the current object
graph already has complete coverage.

## 9. Source and citation preservation

An external source may be scientifically valuable as support, contradiction,
historical context, an abandoned direction, a benchmark comparator, or a
falsifier. The fact that a source is not used by the current implementation does
not remove its provenance value.

Each future source record should preserve, when known:

- stable source ID;
- title, author or organization, and publication date;
- canonical URL, DOI, standard number, repository commit, or release identity;
- date accessed;
- the exact GLYPH question or claim for which it was collected;
- classification: `CURRENT_SUPPORT`, `HISTORICAL_CONTEXT`,
  `CONTRADICTORY_EVIDENCE`, `NEGATIVE_RESULT`, `COMPARATOR`,
  `UNUSED_CANDIDATE`, or `SUPERSEDED_SOURCE`;
- quotation boundaries and licence/copyright constraints;
- local snapshot identity or cryptographic digest when lawful and practical;
- reason if the source later becomes unavailable.

A broken URL does not authorize deleting its record. The record changes to
`SOURCE_UNAVAILABLE` and retains the last known metadata and digest.

## 10. Corpus and benchmark provenance

Corpora and benchmark outputs are not interchangeable with code. A benchmark
claim is preserved with its corpus identity, query set, commit, machine,
compiler, parameters, cache state, runtime capability, raw output, and result
interpretation.

Known provenance distinctions must remain visible:

- private or system corpora such as HDFS are evidence for the named scope but
  not automatically publicly reproducible;
- public corpora such as enwik9 support independent reproduction when their
  acquisition identity and preprocessing remain fixed;
- synthetic, tiny, and Pizza fixtures validate bounded properties and must not
  be represented as production-scale performance evidence;
- the documented high-memory boundary of the current implementation remains a
  limitation until a commit-bound measurement supersedes it.

## 11. Fail-closed action policy

Before deletion, destructive replacement, branch retirement, tag movement,
history rewrite, or removal from hosted storage:

1. resolve the exact object, revision, branch, tag, release, corpus, or artifact;
2. compute direct and transitive dependents where registry coverage exists;
3. report unknown coverage explicitly;
4. prepare a non-destructive alternative such as retention, dated movement,
   archival pointer, or additive supersession;
5. obtain the Owner's explicit dated decision if destruction remains proposed;
6. record the result and its commit in the appropriate ledger.

No automated tool may interpret `not currently used`, `already merged`,
`superseded`, `old branch`, `duplicate-looking`, or `zero ahead` as deletion
authority.

## 12. Known gaps and next promotions

At this snapshot:

1. the object graph is explicitly partial and contains only the initial
   governance objects;
2. revision history metadata is not yet fully validated against Git objects and
   commit reachability by the structural checker;
3. the graph's own exact revision needs an external non-recursive closure
   record;
4. the Centenary structural checker is not yet part of the required CI graph;
5. the two old GitHub releases are not immutable and contain no attached
   evidence assets;
6. external citations do not yet have one machine-readable source ledger;
7. the branch and tag inventory is not yet protected by a dedicated tag/archive
   ruleset;
8. independent review counts must remain evidence-based and may not be inferred
   from the existence of this register.

The next safe promotion is to bind this register as one object in
`GLYPH_OBJECT_GRAPH_V1.json`, then incrementally register the proof/evidence
spine. That work must strengthen the existing V1 graph rather than create an
unnecessary clone or V2.

## 13. Closure statement

This register establishes discoverability and preservation intent for the
observed material. It does not close all 247 facets and does not authorize any
deletion.

    INVENTORY RECORDED; COVERAGE PARTIAL; UNKNOWN FAILS CLOSED

