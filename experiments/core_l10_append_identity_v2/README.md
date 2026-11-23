# CORE-L10 — Append Identity V2: nonnormative isolated identity fixture

**Status: research only.** This is a deliberate continuation inside the existing [CORE-L10 Issue #28](https://github.com/yasha1971-coder/glyph-engine/issues/28) and [Draft PR #29](https://github.com/yasha1971-coder/glyph-engine/pull/29). It does **not** change historical Composition V1, the real RLB2/RLR2/Elias–Fano leaf format, PR #27, main, frozen releases, CONTROL or OVH. **No extra Issue or PR is needed.**

## Architecture chosen

A 128-bit immutable `vault_id` identifies the vault. An unsigned 64-bit, never-reused, append-order `doc_id` is the *logical document* identifier, **not** a sorted path, physical inode, content hash or leaf ID. Each changed set of source bytes gets a monotonic `revision`, whereas a rename records only a path-metadata event and does not modify the byte revision. A second logical document with identical content obtains its own `doc_id`, though it can reuse the same physical content-addressed object.

Each immutable research snapshot contains a canonical event prefix, visible document projection, complete expected `(doc_id,revision)` coverage, prior root digest and a domain-separated `snapshot_id`. A snapshot gets its own SHA-256 root pin. Query coordinates are **`[doc_id, revision, byte_offset]` under a specific root SHA**. `PROVEN_EMPTY` is permitted only for a complete source-byte scan of all visible documents in that exact pinned snapshot. If missing, malformed, partially enumerated or unpinned, the independent checker returns `UNTRUSTED` / nonzero exit. **An unsigned pin is not an author signature, secure timestamp, rollback/fork protection or trusted latest-root witness.**

The reference producer uses a temporary **plaintext byte-object fixture** for identity testing. This is **not** a compressed GLYPH package; it must not be counted as a real self-index or credited with RLB2/EF search performance. The underlying PR #29 already tests real frozen RLB2/RLR2/EF indexed leaves. The bridge from these V2 IDs/revision/path bindings to the real indexed leaves is a **separate required Gate B**, not passed by this fixture. No historical `composition_root_id` / `source_manifest_id` is reused by name: the new schema is `GLYPH_APPEND_IDENTITY_V2_RESEARCH`.

## Producer and independent verification

- `identity_v2.py` creates small immutable source-oracle objects and append-only event histories. It validates raw relative path bytes (NUL, traversal, absolute, empty components and backslashes are rejected), IDs, revisions and path collisions. Roots are written via exclusive temp files plus file/directory `fsync`, hardlink publication and a single-writer `flock` + expected-current-root compare-and-swap. A failure before pointer publication cannot convert an incomplete snapshot into the current root. This is *research* Unix local filesystem behavior, **not** complete power-loss crash consistency on all filesystems or multi-writer replication.
- `independent_verifier_v2.py` is a separate standard-library implementation with **no imports from the producer**. It independently replays each event, rechecks the exact committed projections and full history prefix, validates all source-oracle SHA-256 bytes, verifies source-query count/complete canonical coordinates, and rejects forged `PROVEN_EMPTY` or incomplete coverage. The external expected pin list is supplied separately from the bundle. It uses full byte scans, **not** an FM-index proof.
- `gate_v2.py` runs deterministic successive snapshots and the independent verifier against correct and hostile inputs, including re-pinned *structurally inconsistent* root files. The gate imports producer *only for its test construction and corruption injection*; the verifier source never does.

## Seven sequential snapshots

| Snapshot | Change | Required invariant |
|---|---|---|
| V0 | Ingest `z.bin` as doc 0, bytes `banana\0` | ID 0 / rev 0 |
| V1 | Ingest lexicographically earlier `a.bin` as doc 1 | ID 0 remains unchanged |
| V2 | Rename `z.bin` to `q.bin` | ID 0 / rev 0 and bytes unchanged |
| V3 | Change bytes of doc 0 to `new\xffbanana` | ID 0 / rev 1; older rev 0 remains replayable |
| V4 | Ingest a copy of doc 1 as `copy.bin` | New ID 2 but same physical content SHA |
| V5 | Ingest an empty `empty.bin` as doc 3 | Empty document is real and independently hashed |
| V6 | Tombstone doc 1 | Old snapshots retain doc 1; current visible query scope excludes it |

The research fixture uses a fixed 128-bit vault ID **only to keep receipt hashes reproducible**. Production must generate it securely. Path rename/inferred file identity is not determined from matching bytes: an explicit doc ID and operation are required. An ambiguous rename or replacement during actual filesystem watching needs a human/ingest-policy decision.

## Test contract and likely falsifiers

The CI requires seven pinned roots, **49 independently replayed exact queries**, all source identity and root-prefix checks and at least **35 independently rejected hostile cases plus 14 producer rejections**. These include forged empty results, wrong counts/coordinates/revisions, omitted/duplicated coverage, changed roots, modified/missing/symlinked objects, re-pinned wrong event schema/ordinal/history/projection, traversal/NUL/duplicate names, invalid revision, CAS conflict and lock contention. An uncommitted orphan object does not advance `CURRENT` and can be separately cleaned after confirming immutable roots.

Failure modes worth testing later: concurrent rename and ingest observation, actual signed root witness and anti-fork/rollback, atomic *file+receipt* transaction, original paths across multiple OS namespaces, crash recovery on real filesystems, 64-bit/multi-terabyte source indexing, actual RLB2 leaf revision binding, multi-document full-index query evidence and true `Composition V2` normative adoption. **Historical Composition V1 remains frozen.**

## Current scope and next bounded Gate B

No arbitrary-path release is justified by this fixture. Once this gate passes in CI, connect `(vault_id,doc_id,revision,source_sha256)` with a real immutable RLB2/RLR2/EF leaf **whose own mandatory physical hashes are independently verified**. A rename should alter only the metadata root, while revision change should create a new physical leaf and preserve the prior one. Execute a separate source-byte verifier on every reported match and complete coverage across all committed leaves. Use the same existing Issue #28 and PR #29, not a new tracker per step. Retained package/index/proof and temporary disk must all be accounted separately.
