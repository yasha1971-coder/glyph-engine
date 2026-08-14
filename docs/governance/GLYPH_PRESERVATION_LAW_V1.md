# GLYPH Preservation Law V1

Status: **OWNER DIRECTIVE — ACTIVE**  
Effective date: **2026-08-14**  
Owner: **yasha1971-coder**

## 1. Governing rule

No durable part of the GLYPH project may be deleted, purged, rewritten out of
history, or replaced in a way that destroys its prior meaning without an
explicit decision by the Owner.

This authority belongs only to the Owner. It is not delegated to an AI agent,
automation, maintainer, reviewer, contributor, CI job, migration script, or
release process by default.

## 2. Scope

The rule covers tracked source, specifications, tests, evidence, benchmarks,
schemas, documentation, issues, pull requests, releases, tags, and durable
branches. Ephemeral compiler outputs, caches, temporary files, and reproducible
generated build directories are not durable project records.

Removing a statement and inserting a contradictory statement at the same path
is a replacement and is governed as deletion. Deprecation is not deletion.

## 3. Permitted additive change

New material may be added. A better implementation or specification may be
introduced beside the earlier version. The earlier version remains available
and is marked with an explicit status such as `SUPERSEDED`, `HISTORICAL`, or
`ARCHIVED`, together with a pointer to the newer version.

No claim of improvement is sufficient by itself. The new variant must state
which measurable property is better, which compatibility boundary it changes,
and what evidence supports the claim.

## 4. Permitted movement

Material may be moved without changing its meaning. Every important movement
must:

1. preserve Git history where technically possible;
2. leave a pointer or migration note at the old conceptual location;
3. be recorded in `GLYPH_MOVEMENT_LEDGER.md` with an ISO date, old location,
   new location, reason, commit, and approving authority;
4. preserve immutable tags and published evidence identifiers.

## 5. Security or legal emergency

If a secret, personal data, malicious payload, or legally prohibited material
is discovered, automation must stop and notify the Owner. History rewriting or
purging still requires the Owner's explicit decision, except where the hosting
provider independently enforces a legal or security action outside the
project's control. Such an external action must be recorded afterward.

## 6. Supersession protocol

A future `V2` may supersede this law only through a new, additive document
approved by the Owner. `V1` remains in history. The superseding document must
name this document, state the effective date, and enumerate every changed rule.

## 7. Object identity and dependent relations

This law governs identified logical objects and their immutable revisions, not
pathnames alone. The normative identity and relationship model is
`GLYPH_OBJECT_IDENTITY_AND_DEPENDENCY_V1.md`.

Before an important move, supersession, destructive replacement, or proposed
deletion, the exact target identity and its transitive incoming dependency
closure must be recorded. Unknown registry coverage means `UNKNOWN`, not “no
dependents.” Any unresolved dependent blocks deletion. Pure movement preserves
the logical object identity and records a new location binding; normative byte
changes create a new immutable revision identity.

The logical identity of this law is:

    glyph-object:governance:preservation-law:v1

Its first published revision is bound to Git commit
`dbc16e72a96f664301499b5fad53a0aa7f170895` and the original SHA-256 digest is
`55550dc429f0a5d514472969682a7fd5a165c9c707a9dcd9d2a60a0000200a1e`.
Later revisions must remain linked in the object graph.

