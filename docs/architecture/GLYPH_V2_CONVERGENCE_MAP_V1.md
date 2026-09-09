# GLYPH V2 Convergence Map V1

Status: integration map; not a product claim  
Date: 2026-09-09  
Integration branch before this change:
`glyph-v2-1tb-truth-gate` at
`17b59df03bda8c3a9eb9ac027a59dd5f453824d6`

## 1. Purpose

This document defines how previously verified, measured and experimental GLYPH
lines may converge into a growing Personal Memory without changing the public
meaning of GLYPH or silently promoting research targets into current
capabilities.

The architectural target is broader than the currently verified product. The
verified core remains narrow:

> deterministic, replayable exact-byte retrieval evidence over an identified,
> committed corpus state.

Personal Memory is an append-only repository of such immutable states. Adding
data creates a new generation; it does not mutate the meaning or coordinates of
an already published generation.

## 2. Status vocabulary

| Status | Meaning |
|---|---|
| `VERIFIED` | A named executable gate and committed evidence support the exact claim. |
| `MEASURED` | A named run measured the property for identified artifacts; it is not universal. |
| `CODE-PROVEN` | The committed implementation or checker establishes a bounded property. |
| `EXPERIMENTAL` | Useful implemented research, outside the required public V1 chain. |
| `SPECIFIED` | A contract exists, but the end-to-end implementation is not complete. |
| `TARGET` | Intended architecture; no present capability claim. |
| `UNVERIFIED` | A prior statement lacks a located receipt sufficient to repeat the claim. |

No lower status may be described publicly using a higher-status verb.

## 3. Public-face compatibility

| Public surface | Observed public boundary | V2 compatibility rule |
|---|---|---|
| `https://glyph.rs/` | Verifiable exact-byte retrieval over committed corpora; P/R/O verified; Embedded I0 pre-freeze; production not claimed. | Keep this as the stable verified kernel. V2 storage and Personal Memory remain an explicitly separate experimental line until their own gates close. |
| `https://demo.glyph.rs/` | Experimental fixed 50,000,000-byte Pizza & Chili corpus; exact UTF-8 bytes; zero-based offsets; at most 20 offsets; no upload; no production claim. | Do not connect the demo to arbitrary Personal Memory intake or present it as a general archive product. |
| `https://glyph-evidence-duel.yasha1971.chatgpt.site/#top` | One enwik8 instance at `0.987100010x`; exact count and locate; extraction, cross-corpus matrix, fixed-RAM guarantee and TB result not claimed. | New V2 work must not imply that chunk inventory is compression, that unpack is closed for the RLBWT runtime, or that one ratio generalizes. |
| GitHub default `main` | V1 proof/runtime/operator chain and Embedded I0 pre-freeze boundary. | Keep `main` stable until a deliberate release decision. Development claims must name the V2 branch and commit. |
| X posts listed in section 4 | Historical questions and bounded claims about SA64/sharding, exact evidence, Embedded review, substring uniqueness and one compact enwik8 counterexample. | Preserve chronology. Later architecture may answer those questions, but must not rewrite them as already proven results. |

There is no contradiction between "fixed corpus" and append-only Personal
Memory if and only if every generation is an immutable committed corpus state.
The repository grows by adding generations and content objects, never by
changing the historical object that an evidence result names.

## 4. Public-post claim ledger

These rows record the text recoverable from the official X embed response. An
ellipsis in an embed is not treated as read text.

| Date | Post | Recoverable claim boundary |
|---|---|---|
| 2026-05-28 | `2060139266018095140` | Reported 1,061 clones / 418 unique over 14 days and asked for experience with production exact indexes above 4 GB, SA64 versus sharding and routing overhead. This is a request for evidence, not a scalability result. |
| 2026-06-23 | `2069439669931040941` | Positions GLYPH as a tool for independently verifying an exact string in a fixed archive. |
| 2026-07-19 | `2078637422418526363` | Requests independent review before freezing Embedded C ABI V1 and explicitly says implementation had not started. |
| 2026-08-16 | `2089041663327130089` | Reports a 50,000,000-byte text experiment over 100,000 positions, median unique length 13 bytes, with 48.006% still non-unique at the embed's truncated boundary. |
| 2026-08-28 | `2093335860959654364` | Claims exact count, byte offsets and reproducible evidence for one counterexample to a second corpus-sized index; asks for the workload where it matters. The official embed truncates the remainder. |

## 5. Binding architecture decisions

### D-005 — Hybrid storage

Use a compressed self-index only where corpus structure and complete runtime
overhead justify it. Other classes may use content-addressed deduplication,
delta encoding or class-specific lossless codecs. Routing is a physical choice;
it must not change logical corpus identity, document order, canonical
coordinates or restore bytes.

### D-007 — Composition before Index Forest

Index Forest runtime remains blocked until composition semantics and evidence
are accepted. Preserve:

- partition-independent `composed_corpus_id`;
- partition-dependent `composition_root_id`;
- canonical `(doc_id, doc_offset)` coordinates;
- `block_id` only as execution and coverage provenance.

`GLYPH_COMPOSITION_FOUNDATIONS_V1` is additive. It does not by itself approve
Index Forest, atomic publication or distributed operation.

### D-014 — Append-only Personal Memory

Personal Memory is a growing repository. A write creates a new immutable
generation that references content-addressed objects. Historical generations,
evidence identities and their byte coordinates remain addressable.

### Human authority boundary

Human policy decides what enters the repository, what may be deleted, what is
published and which generation is authoritative for a human task. Models may
interpret authenticated evidence but cannot authoritatively decide exact
presence, provenance, currentness, supersession or verified history.

Invariant:

    MODEL OUTPUT != VERIFIED STATE

## 6. Target capability ledger

| Capability | Current state | Integration requirement |
|---|---|---|
| V1 authentication/evidence | `VERIFIED` for the named V1 P/R/O chain; signing remains unimplemented. | Preserve the required chain as the trust kernel. |
| Composition Reference V1 | `VERIFIED` executable reference; its branch head is already reachable from V2. | Integrate Foundations/Evidence without changing V1 wire fields silently. |
| Binary-safe RLBWT V2 | `MEASURED` and hostile-gated for the named enwik8 artifacts. | Keep corpus/hash/parameter scope attached to every ratio and latency. |
| Compact/lazy Rank+Locate | `EXPERIMENTAL`; compact structures exist, but there is no general bounded-RAM theorem. | Define startup/RAM budgets and measure complete packaged runtimes per corpus class. |
| RLB3X compactness | `EXPERIMENTAL`. | Treat it as one candidate engine selected by measured class fit. |
| Personal Vault ratio `0.335` | `UNVERIFIED`: no supporting receipt was located in the inspected repository, handoff or public surfaces. | Require corpus identity, source bytes, complete included-byte numerator, parameters, artifact hashes, command and independent restore before publication. |
| Content-addressed chunks | `CODE-PROVEN` only for restartable source-chunk hashing and checkpoint validation in the V2 truth gate. | A source hash manifest is not yet a payload object store. |
| Dedup/delta/class codecs | `TARGET`. | Measure a baseline first; preserve byte-perfect independent restore. |
| Incremental immutable segments | `SPECIFIED/TARGET` across prior segment and Personal Memory work, not one closed V2 repository protocol. | Define generation publication, crash recovery, retention and reader concurrency. |
| Byte-perfect restore | `VERIFIED` in bounded prior workflows; not yet a full Disk D/V2 container result. | No source deletion before independent cold recovery and replica/witness gates. |
| Bounded startup/RAM | `MEASURED` for named instances only. | Do not claim a fixed bound until the format and gate enforce it. |
| Heterogeneous class router | `TARGET`. | Routing must be deterministic, recorded, replayable and semantically invisible. |

## 7. Branch convergence inventory

Remote heads were read with:

    git ls-remote --heads origin

Reachability was checked against
`17b59df03bda8c3a9eb9ac027a59dd5f453824d6` using
`git merge-base --is-ancestor`.

`reachable` means Git history is present in V2. It does not by itself mean that
the branch's idea is obsolete or that every later branch-only refinement is
present.

| Branch | Head | Relation to V2 base | Role |
|---|---|---|---|
| `glyph-v2-1tb-truth-gate` | `17b59df03bda8c3a9eb9ac027a59dd5f453824d6` | active base | Current integration line. |
| `main` | `9f79fd5cf8f969ed46c2c1ad7945f6b5a2944edb` | reachable | Public stable V1 line; default and protected. |
| `rlbwt-binary-safe-v2` | `c33cb493bc75f13ee4a904fabb190a29a196c94b` | reachable | Binary-safe compact self-index source. |
| `glyph-v1-unified` | `81cacfcbc98408241c4a5429845809514b01d880` | reachable | Portable V1 trust kernel source. |
| `personal-vault-v0-runtime-v2` | `c2a847968886d3da9b45a85f62cd338e85ca837c` | reachable | Closed-loop Vault runtime source. |
| `composition-reference-v1` | `81628f843dc8aa633c65594a66d2386bfc3cb5c2` | reachable | Executable Composition Reference V1. |
| `composition-independent-replay-v1` | `81628f843dc8aa633c65594a66d2386bfc3cb5c2` | reachable; identical head | Duplicate pointer retained for provenance. |
| `composition-semantics-v1-spec` | `5e67d0a3178581414356aa5c12e217e93d7abcea` | reachable | Earlier composition contract history. |
| `feature/segmented-v0.2` | `dc124b23e3f2b0e31fed654e0241fe9010aef21f` | reachable | Historical segmented runtime research. |
| `embedded-api-v1` | `ff7ab6b30a84e4691dc86d61a6d65e76d4b2a0df` | reachable | Embedded contract history. |
| `operator-path-v1` | `8e8019b42a9c23e8d0b9d1c81ab815503952f877` | reachable | Verified operator path history. |
| `public-surface-v1` | `192307b5a800ae3567ff036bbfc1fc5e34f091a4` | reachable | Public-site source history. |
| `runtime-conformance-v1` | `77aaf82d75d9fba848ee14a7c34426b37103da3a` | reachable | Verified runtime history. |
| `site-public-surface-v1` | `9f79fd5cf8f969ed46c2c1ad7945f6b5a2944edb` | reachable; identical to `main` | Duplicate pointer retained for provenance. |
| `composition-foundations-v1` | `c6c9be67033c09de8deef7003e31049b5187c10a` | diverged; two unique commits | Required D-007 specification input; integrate before further forest work. |
| `centenary-247-v1` | `e2f37209301ef09bfd097c90352bf71f436ad338` | diverged; 13 unique commits | Governance/preservation input; separate review surface. |
| `embedded-i0-second-review-v1` | `dff6127b0758661c8cf36456dbd4e40a671122c6` | diverged; two unique commits | External-review input; do not silently declare ABI frozen. |
| `personal-vault-v0` | `add9430c5aaa6f6fe45aa93c5589f974407b5190` | diverged; three unique commits | Corpus fixture/workflow; open draft PR may be superseded by runtime V2 and requires disposition. |

No branch deletion is authorized by this map. Before eventual deletion, preserve
the unique head with an immutable annotated tag or merged history and record its
replacement and proof boundary.

## 8. Tag policy

The 13 current remote tag names are historical evidence anchors and must not be
moved or deleted:

- `composition-executable-reference-v1`
- `evidence-bundle-v1`
- `glyph-binary-runtime-v1-verified`
- `glyph-embedded-i0-second-review-v1`
- `glyph-operator-path-v1-verified`
- `glyph-proof-graph-v1-verified`
- `retrieval-v1`
- `rlbwt-bounded-evidence-v1`
- `sentinel-safe-v1`
- `structural-fingerprint-v0-verified`
- `v0.1`
- `v0.1-alpha`
- `v0.1-stable`

## 9. Converged layer model

1. Human-authorized intake records an exact source snapshot.
2. Content truth binds ordered files and fixed chunks.
3. Immutable generation publication names a stable corpus state.
4. Lossless physical routing selects self-index, dedup, delta or class codec
   without changing semantic identity.
5. Independent restore reconstitutes exact source bytes from stored payloads.
6. Composition maps immutable segments into one ordered logical corpus while
   retaining a distinct physical root.
7. Exact query engines return canonical `(doc_id, doc_offset)` coordinates.
8. Authentication and portable evidence bind query, corpus, runtime, result and
   replay.
9. Models and user interfaces interpret verified state but cannot replace it.

## 10. Ordered next gates

1. Integrate the two unique Composition Foundations commits into V2 without
   enabling Index Forest.
2. Run the complete existing verification chain and the 16-test Personal Memory
   component suite.
3. Define `GLYPH_CONTENT_OBJECT_STORE_V1` only after a measured dedup baseline.
4. Build the smallest payload container that independently restores every byte
   from object hashes; do not add search yet.
5. Define crash-safe immutable generation publication and replay.
6. Add the deterministic heterogeneous router after restore semantics are
   closed.
7. Attach Composition Evidence to generations before any Index Forest runtime.

Disk D inventory must not be repeated merely to exercise these gates. The
existing result remains inventory evidence, not compression evidence.
