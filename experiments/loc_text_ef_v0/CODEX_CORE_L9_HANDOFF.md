# CODEX EXECUTION CONTRACT — GLYPH CORE-L9-A/B

**Status: TASK SPECIFICATION ONLY. No Codex task has been launched by adding this file.**

**Owner:** human engineer (strategic veto). **Architect/reviewer:** ChatGPT. **Implementer:** Codex Cloud or Codex CLI in an authorized isolated worktree.

**Repository:** https://github.com/yasha1971-coder/glyph-engine
**Prerequisite research branch:** \`codex/glyph-lf-bound-loc3-ef-research-20261105\`
**Parent:** [Draft PR #21](https://github.com/yasha1971-coder/glyph-engine/pull/21)
**Issue:** [CORE-L9 #24](https://github.com/yasha1971-coder/glyph-engine/issues/24)
**Frozen source, DO NOT MUTATE:** \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`
**Predecessor evidence:** [CORE-L8 100MB verified-before-publication + offline replay CI PASS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42144629682).
**Existing L8 code:** \`experiments/loc_text_ef_v0/trusted_materialize_v0.py\` and \`independent_materialization_replay_v0.py\`.
**Existing exact query:** \`experiments/loc_text_ef_v0/query_ef_research.py\`, \`rank_v2_ef_runtime_gate.py\` and \`self_index_exact_range_extractor.py\`.
**Original preserved V0 proof source:** \`experiments/personal_vault_v0/proof_carrying_recall_v0.py\`, \`verify_proof_carrying_recall_v0.py\` and \`proof_carrying_recall_gate_v0.py\`.
**Original pinned fixture generator:** \`tools/check_rlbwt_query_v2.py\` at frozen commit.

## Persistent engineering GOAL

Deliver **one independent, reproducible, fail-closed exact-search evidence capsule** binding:
\`a caller-trusted root SHA-256 -> pinned RLB2/RLR2/physical EF -> complete exact byte search -> count + canonical corpus offsets + verified excerpts -> a deterministic query receipt -> a separate independent verifier\`.

A successful query is not automatically a truth claim, a proof of what happened before a trusted root, a signature, a full multi-document composition proof, or a production release.

**Completion criteria are executable results, not prose or attractive metrics.** If a proposed property cannot be verified against the actual code and hashed artifacts, retain it as HYPOTHESIS.

## Execution protocol for Codex

1. **READ-ONLY reconnaissance first.** Verify the precise current HEAD of the prerequisite branch and all referred files/CI; inspect the existing schemas and API signatures. Record observed facts and any source conflicts. The branch might have moved since this specification was committed; never assume a stale SHA.
2. Work on a **single new isolated branch/worktree**, e.g. \`codex/core-l9-exact-query-proof-a-b\`, created from the actual prerequisite branch HEAD. Do not edit or reset a previous frozen ref. Do not modify branches \`main\`, \`rlbwt-binary-safe-v2\` or public deployment settings.
3. Implement **A then B**, with exhaustive small tests and hostile tests. Build or adapt existing helper modules instead of rewriting FM/BWT/EF implementations. Keep changes restricted to a new \`experiments/core_l9_exact_query_v0/\` subtree plus its own non-invasive CI workflow. Do not silently change code/files in \`experiments/loc_text_ef_v0/\` unless an independently demonstrated bug blocks the work; when that happens, report the defect and stop for an architectural decision or add an isolated adapter rather than mutating frozen behavior.
4. Make tests run in GitHub Actions, pinned to actual frozen reference \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`, not local unversioned software. Report the CI URL, exact HEAD, original source SHA, source search and receipt sizes, RAM, time and complete negative tests.
5. Open a **Draft stacked PR targeting** \`codex/glyph-lf-bound-loc3-ef-research-20261105\` when implementation and at least the small fixture tests PASS; no squash, merge, rebase, tag, release, OVH, public demo, user-data ingest, C ABI freeze or production modification. Preserve ability to compare/replay the original prerequisite branch.
6. If any prerequisite is missing, or any proof condition fails, return **BLOCKED** with the narrowest observed falsifier. Do not label skipped/pending tests as PASS. Do not use hidden build artifacts or reintroduce a second retained source archive.

## Exact proposed implementation boundary: L9-A

Use the original frozen binary-safe fixture:
\`b"\x00banana_bandana\xff\x00banana!"\`, 24 source bytes.

Use the actual RLB2+RLR2+EF search runtime already present. Generate initial SA/BWT for construction only, build and hash the physical search artifacts, and then **remove raw source and construction SA/BWT before query execution**. Test oracle may independently reconstruct the source and compare its SHA to the pinned source identity; it must not secretly serve the source bytes as an input to the search implementation.

Support **non-empty exact byte strings**, with patterns including NUL, 0xFF, binary adjacency \`ff 00\`, short repeated matches, beginning/end boundary, pattern longer than corpus, absent pattern, capitalization, one-space versus two-space, empty source query rejection, and candidate offsets crossing text-sample boundaries.

For each query:
- Use the real FM-backward-search count and full locate (not a fabricated count read from oracle).
- Compare against an **independent** byte scan of the fully source-SHA-verified corpus; verify complete list of **sorted canonical corpus byte offsets**, not just membership. The output presentation can be sorted after raw locate if the exact documented API semantics require it.
- Never infer completeness from a capped \`max_offsets\`. Distinguish \`full=true\` from \`limited\` and \`count\` from \`len(returned_offsets)\`.
- For each reported offset, independently extract source-byte excerpt from physical EF+RLB2/RLR2 and check byte identity, length and SHA. Do not use a separately retained source payload for native query/excerpt.
- Accept \`FOUND\` only for nonempty verified match(es) and complete evidence. Accept \`PROVEN_EMPTY\` only for **full, exact, independently verified count=0 over this single complete committed corpus** when the required source/root/rank/locate coverage checks pass. Any missing/unpinned mandatory component, malformed index, incomplete corpus, wrong root, stale scope claim or verifier mismatch becomes \`UNTRUSTED\`, NEVER \`PROVEN_EMPTY\`.
- For single-document test scope, \`corpus_byte_offset\` is the actual coordinate. **Do not manufacture** \`document_id\` from corpus positions. To emit \`(doc_id,doc_offset)\`, first explicitly commit and independently verify the document mapping against existing Composition normative contracts in a later L9-C gate.

## Query receipt: L9-B

Propose a **canonical JSON byte serialization** with a declared version and no duplicate keys, unexpected fields, nonfinite numbers, invalid UTF-8, wrong lengths, or truncated manifests. The root is external (no authority claim). A receipt must contain at least:
- schema version, exact external root digest, pinned source identity/length, all index component digests/lengths and query-runtime identity/scope;
- non-empty exact \`pattern_hex\` and its exact byte length, full/partial search policy;
- exact \`count\` from the real index and **full** sorted list of corpus byte offsets for small fixture;
- each excerpt's exact source offset, length and SHA256 (and whether excerpt bytes are included);
- status \`FOUND\` / \`PROVEN_EMPTY\` / \`UNTRUSTED\`, with reasons/scope, no semantic assertions.
- bound the receipt size. When count/offset enumeration exceeds budget, return an explicit **LIMITED/UNTRUSTED for completeness** state and preserve count separately; never fake a complete list.

Build a **separate independent verifier executable/file**, no import of the receipt producer. It must hash the input root and complete physical state, reconstruct or independently use a *verified* full source snapshot for an exact byte scan, compare exact count, every offset, every excerpt digest and any reported empty result. Refuse wrong count, missing hit, added fake hit, duplicate/unsorted offset, changed \`pattern_hex\`, excerpt/root/source SHA mutations, forged \`PROVEN_EMPTY\`, omitted files and altered query policy. This verifier may read the L8 materialized source **only after** checking its pinned SHA, and must record its full source-read cost; this is not a succinct cryptographic proof.

**Independence check:** producer code/import graph and verifier code/import graph are separate. Shared immutable schema constants or a tiny pure serialization helper must be justified; ideally keep verifier standalone to detect equivalent defects.

## Required gates and stop conditions

- Test cases cover all \`24 * 25 / 2 = 300\` **nonempty source substrings** (count duplicate byte values separately as case coverage), plus ≥10 adversarial/absent patterns and empty input rejection, and test at least one match crossing a text-sample boundary.
- True \`count\` and complete offsets are identical to the independent byte oracle for all positive and negative cases; **no acceptance with incorrect offsets**.
- ≥12 explicitly enumerated hostile receipt and component mutations produce \`UNTRUSTED\` and exit nonzero, including three independent \`PROVEN_EMPTY\` forgery attempts. Missing rank or EF must never produce empty.
- Full SHA256 source identity, root and frozen component identity are independently checked before query receipt is allowed to claim verified coverage; source/historical input mutation is rejected.
- No production use of \`PROVEN_EMPTY\` on partial coverage, on unknown doc mappings or across multiple corpus blocks. Report exactly what claim was proven on the frozen single-file fixture.
- Record receipt size and complete verification work (source read bytes, SHA operations, CPU wall, peak memory). Record all failures, repairs and attempts; **no CI-pass laundering**.
- Two independent CI jobs if possible: producer/exact fixture, independent hostile replay. Keep all outputs below an explicit bound and avoid large corpus/index uploads.
- **After A+B PASS** and separate owner review, one follow-up 100MB enwik8 exact-query candidate may be proposed as L9-D; do NOT silently escalate to multi-GB/TB, signed roots, public server or Index Forest.

## Competitive scientific honesty

The existing run-LF identity proof in Mathlib (\`RunIntervalLFProof.lean\`) certifies a **known abstract mathematical property**, not this query verifier. Do not call this a Lean-proven C++ implementation. Existing succinct index approaches (FM/r-index/Move-r/Orbit), Merkle proofs and authenticated search have prior art. Any "first in the world" or "orders of magnitude faster" claim requires **independently pinned comparable systems, identical work and package accounting**.

## Requested final Codex report

Return:
- \`STATUS = VERIFIED_SMALL / BLOCKED / RESEARCH_ONLY\`;
- branch, exact commit SHA, Draft PR URL, all CI URLs/statuses;
- precise count/offset/byte-excerpt test totals and hostile rejection totals;
- one example deterministic **FOUND** and one fully scoped **PROVEN_EMPTY** receipt, with external root SHA and caveats;
- CPU/RAM, retained archive bytes, temporary disk, receipt bytes, verification workload;
- open gaps: compositional doc coordinates, evidence O3/O4, signatures/currentness witness, rollback/fork, 64-bit/TB, C ABI;
- \`FALSIFIERS/FAILED_PREDICTIONS\` explicitly preserved.

**Owner strategic veto:** implement and measure only. Never merge, freeze, change main, modify OVH, publish release or claim production without explicit owner authorization.
