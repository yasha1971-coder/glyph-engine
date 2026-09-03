# GLYPH Personal Vault Repository Architecture V0

Status: design baseline for real-device pilot; not a frozen on-disk format.

## Goal

Make GLYPH a local-first personal memory store where original files may eventually be removed after verified ingest, while users continue to browse logical folders, restore bit-perfect originals, search through GLYPH, and use an AI planner without allowing AI-derived state to mutate canonical evidence.

## Design influences reviewed

- Kopia: repository abstraction, content-addressed storage, blob backends, manifests, mountable snapshots, maintenance and verification.
- restic: immutable pack/blob/snapshot model and explicit repository/data integrity checks.
- Borg/Vorta: mountable archives, append-only repository operation, integrity verification, GUI that hides terminal complexity.
- Perkeep: immutable content-addressed blobs with mutable human-level objects represented as claims above immutable storage.
- IPFS/UnixFS: content-addressed directory DAGs and independently verifiable subtrees, while noting that canonical identity requires frozen construction parameters.
- Tahoe-LAFS: erasure-coded resilience and verification of subsets using hashes/Merkle structures; retained as a future durability layer, not V0.
- Dropbox/OneDrive-style placeholders: user-visible files can remain in the namespace while physical bytes are absent until materialized.
- Apple File Provider: dataless versus materialized local items; preferred future iOS integration model.
- Apple FSKit: future macOS user-space filesystem integration.
- Windows Cloud Files / WinFsp: future placeholder or user-space filesystem integration on Windows.
- FUSE: future Linux mount layer.

## Core architectural decision

Do NOT store one `.glyph` archive per source file and do NOT use one monolithic lifetime `.glyph` file.

Use one logical Vault backed by multiple immutable segments plus manifests and object metadata.

Physical repository sketch:

```
GlyphVault/
  repo.meta
  manifests/
    roots/
    snapshots/
  segments/
    00000001.glyph
    00000002.glyph
    ...
  objects/
    object-map.*
  derived/
    text/
    metadata/
    ai/
  journal/
  cache/
  quarantine/
```

The exact filenames/formats are provisional.

## Canonical state versus derived state

Canonical state:

- original object bytes
- object content hash
- logical path at ingest
- observed filesystem metadata captured at ingest
- object-to-segment mapping
- immutable segment identity
- committed manifest root

Derived/disposable state:

- extracted PDF/DOCX/XLSX text
- OCR
- transcription
- AI descriptions/tags/entities
- embeddings
- caches
- alternative virtual views

Invariant: derived state may be deleted and rebuilt without changing canonical object identity.

## Object identity and logical namespace

A physical object is identified by content identity, not by pathname.

A logical path is a view/reference to an object version.

The same object may therefore appear in multiple virtual views without storing its bytes multiple times:

```
Original folders/2009/Italy/contract.pdf
By date/2009/contract.pdf
By type/PDF/contract.pdf
Versions/contract/2009-07-15.pdf
```

## Immutable segments

V0 should grow by sealing new segments instead of rewriting an ever-growing global file.

Target segment size is a tunable engineering parameter, initially on the order of hundreds of MiB rather than one object per segment or one vault-wide segment.

Reasons:

- bounded corruption blast radius
- atomic publication
- incremental ingest
- compaction can be explicit and recoverable
- parallel verify/restore
- future replication/sync at segment granularity
- easier migration and format evolution

## Manifest root

Every committed Vault state has a small root manifest that binds:

- repository format version
- segment identities and hashes
- object map identity
- logical namespace/snapshot identity
- derived-state generation identifiers separately
- parent manifest when a new generation is committed

Publishing a new root is the commit point.

## Ingest state machine

Original deletion is never part of initial ingest.

```
DISCOVERED
 -> CAPTURED_METADATA
 -> INGESTING
 -> OBJECT_HASHED
 -> SEGMENT_WRITTEN
 -> MANIFEST_PREPARED
 -> RESTORE_TESTED
 -> HASH_EQUAL
 -> COMMITTED
 -> SCRUBBED
 -> ELIGIBLE_TO_FREE_SOURCE
 -> SOURCE_QUARANTINED
 -> SOURCE_REMOVED
```

Any failure before COMMITTED leaves the source untouched.

`ELIGIBLE_TO_FREE_SOURCE` is a recommendation state, not automatic deletion.

## Free-space semantics

The user-facing operation should be `Free original space`, not `delete source`.

It is permitted only when every selected object:

1. is reachable from a committed manifest root;
2. has a successful bit-perfect restore test;
3. restored SHA-256 equals ingest SHA-256;
4. has passed the required scrub policy;
5. has no unresolved repository-health error.

For early pilots, source files should first move to quarantine rather than be permanently deleted.

## Quarantine

V0 real-data pilot: quarantine is mandatory.

Default policy proposal: keep source copies for a user-visible retention window before permanent removal. The exact default is product policy, not a storage-format invariant.

Future stronger policy: allow permanent source removal only after a second independent Vault replica or durable backup exists.

## Verification and scrub

Required commands/API concepts:

- `verify manifest`
- `verify objects`
- `verify segments`
- `scrub --sample`
- `scrub --full`
- `restore-test <object>`

Verification state is recorded separately from canonical bytes; failed verification can never be overwritten by an AI layer.

## Read path

Logical file open:

```
virtual path
 -> object/version reference
 -> object map
 -> GLYPH segment(s)
 -> restore/materialize bytes
 -> cache
 -> host application
```

Search path:

```
human query
 -> AI planner
 -> strict planner protocol
 -> GLYPH exact evidence
 -> object coordinates
 -> found | ambiguous | partial | not_found
 -> AI explanation/clarification
```

## Write/edit path

Mounted Vault V0 should initially be read-only.

Later, editing a materialized file creates a new immutable object/version and publishes a new manifest root. It must not overwrite prior canonical object bytes in place.

## Platform layers

Linux pilot:

- first: normal repository plus CLI/API
- next: FUSE read-only virtual mount

Windows:

- preferred product direction: Windows Cloud Files placeholder/hydration model when appropriate
- alternative/general user-space filesystem: WinFsp
- ProjFS is useful for projected hierarchical data but Microsoft recommends Cloud Files API for cloud-files scenarios

macOS:

- first: application/CLI and materialization cache
- future: FSKit user-space filesystem

Apple mobile:

- iOS/iPadOS: File Provider extension; dataless metadata objects become materialized when opened

## What GLYPH should copy conceptually, not literally

From Kopia/restic/Borg:

- repository abstraction
- immutable packed storage
- snapshots/manifests
- periodic integrity verification
- mount/browse without full restore

From Perkeep/IPFS:

- content identity independent of human naming
- logical human objects/views above immutable content

From Dropbox/File Provider/Cloud Files:

- visible placeholder namespace distinct from physical materialization
- explicit local cache lifecycle

From Tahoe-LAFS:

- future redundancy/repair thinking; do not add erasure coding to V0 yet

## What NOT to copy

- backup-product UX as the final model: GLYPH is intended as active personal memory, not merely disaster recovery
- cloud dependency as a correctness requirement
- mutable in-place canonical objects
- one huge vault file
- hidden automatic source deletion
- AI metadata as authoritative identity
- filesystem folders as the physical storage model

## V0 pilot scope

Build only enough repository structure to ingest real user files safely:

1. `glyph vault init <path>`
2. `glyph add <files-or-folder>`
3. seal an immutable segment
4. create object map + manifest root
5. restore every ingested object and compare hash
6. `glyph verify`
7. `glyph ask ...` through existing AI-GLYPH bridge
8. `glyph restore <object>`
9. report `safe_to_free_bytes`, but do not permanently delete sources in the first real-data pilot

Virtual filesystem, cross-device sync, erasure coding, cloud backends and automatic compaction remain later layers.

## Primary product invariant

A user must never need to understand segments, BWT, LOC2, manifests or AI probes to trust the Vault.

The visible contract is:

> I put files into GLYPH. It proves it can restore them exactly. I can still browse and find them. I can reclaim source space only after GLYPH proves the stored state is healthy.
