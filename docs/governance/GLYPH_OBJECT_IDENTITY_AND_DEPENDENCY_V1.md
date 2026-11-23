# GLYPH Object Identity and Dependency Model V1

Status: **NORMATIVE DRAFT**  
Established: **2026-08-14**  
Logical identity: `glyph-object:governance:object-identity-dependency:v1`  
Governed by: `GLYPH_PRESERVATION_LAW_V1.md`

## 1. Problem

Preserving a pathname is not enough. A durable object may be depended upon by
specifications, tests, proofs, APIs, evidence, releases, or external users.
Changing, moving, superseding, or deleting it can therefore change other
objects even when their bytes remain untouched.

Every governed action must identify both the target object and the affected
relationship paths.

## 2. Two-level identity

### 2.1 Logical object identity

A logical object has a stable identifier:

    glyph-object:<kind>:<namespace>:<name>:v<major>

The logical ID names the continuing meaning of the object. It remains stable
across a pure move. It does not imply that two revisions have identical bytes.

Required fields:

- `object_id`;
- `kind`;
- `title`;
- `major_version`;
- `authority`;
- `created_at`;
- `lifecycle_state`;
- current and historical locations.

### 2.2 Immutable revision identity

A revision is the exact bytes of one logical object at one publication
boundary. Its primary content identity is:

    sha256:<64 lowercase hexadecimal digits>

The revision record also binds:

- repository identity;
- Git commit identity;
- Git blob identity when applicable;
- path at that commit;
- byte length;
- schema or format version;
- publication timestamp and authority.

Moving an object preserves `object_id` but creates a new location binding.
Changing normative bytes creates a new revision identity.

## 3. Relationship identity

A relationship is a first-class durable record, not an informal hyperlink.
Each edge has a stable `relation_id`, a source object/revision selector, a type,
a target object/revision selector, a normative strength, and evidence.

V1 relation types are:

- `DEPENDS_ON` — source cannot retain its declared meaning without target;
- `IMPLEMENTS` — source realizes the target contract;
- `VERIFIES` — source checks a stated property of target;
- `EVIDENCES` — source supplies evidence for a claim about target;
- `PRODUCES` — source process creates target;
- `REFINES` — source adds precision without replacing target identity;
- `SUPERSEDES` — source is a declared successor while target remains preserved;
- `MOVED_FROM` — current location continues the same logical object;
- `COMPATIBLE_WITH` — compatibility is asserted with named scope;
- `INVALIDATES` — source finding defeats a named claim or revision.

Every edge MUST declare whether it binds a logical object or one exact revision.
Proofs, signatures, evidence, conformance reports, and release approvals MUST
bind exact revisions. Navigation and ownership relationships MAY bind logical
objects.

## 4. Dependency impact closure

Before an important move, supersession, destructive replacement, or proposed
deletion, tooling MUST compute and record:

1. the exact target `object_id` and revision;
2. all direct incoming and outgoing edges;
3. the transitive incoming dependency closure;
4. every evidence, proof, release, API, or external contract bound to the
   target revision;
5. the proposed repair, migration, compatibility result, or explicit break;
6. unresolved dependents;
7. the Owner's dated decision when deletion or destructive replacement is
   proposed.

An unresolved incoming `DEPENDS_ON`, `IMPLEMENTS`, `VERIFIES`, `EVIDENCES`, or
`PRODUCES` path blocks deletion. A `SUPERSEDES` edge never deletes its target.

## 5. Identity-preserving operations

### Addition

Creates a new logical object and its first revision. It may add edges but may
not silently change existing edges.

### Move

Preserves logical `object_id`, records old and new locations, adds a
`MOVED_FROM` edge or equivalent ledger entry, and revalidates path-based
consumers. An unchanged content digest is strong evidence of a pure move.

### Revision

Preserves logical `object_id` only when the governing contract declares the
change compatible. It creates a new revision and retains the earlier revision.

### Supersession

Creates a new logical object or major version, adds `SUPERSEDES`, retains the
target, and lists every dependent that migrated or remains pinned.

### Deletion

Requires an impact-closure artifact and the Owner's explicit dated decision.
Absence from the registry is not proof that the object has no dependents.

## 6. GLYPH identity distinctions

Identity domains MUST NOT be conflated:

- `composed_corpus_id` identifies the ordered logical document sequence and is
  independent of physical partitioning;
- `composition_root_id` identifies one ordered block layout and its manifests;
- index identity identifies a physical index revision built for a corpus;
- result identity binds corpus/root as applicable, query bytes, profile, and
  result semantics;
- evidence identity binds the exact result, generator, verifier, schemas, and
  replay inputs.

A physically identical file is not automatically the same logical object, and
two different physical layouts may implement the same logical corpus identity.

## 7. Registry and fail-closed rule

The initial machine-readable registry is
`GLYPH_OBJECT_GRAPH_V1.json`. Registry completeness is a claim requiring an
audit; missing registration MUST be reported as `UNKNOWN`, never interpreted as
`NO_DEPENDENCIES`.

The structural gate MUST reject duplicate object IDs, duplicate relation IDs,
dangling edge endpoints, invalid revision digests, and missing identities for
the Centenary governance objects.

