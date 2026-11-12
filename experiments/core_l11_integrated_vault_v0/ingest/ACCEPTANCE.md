# Bounded filesystem integration contract (not acceptance evidence)

Status: IN_PROGRESS. Existing Issue #28 / Draft PR #29.
Baseline: 3a81e9839b4f9a2959efa8c56364a2996c2124d6.

Reuse CORE-L9 QueryState for each physical RLB2/RLR2/EF object and
CORE-L8 trusted_materialize_v0.py for independent selected-file restoration.
Reuse fixture.py construction geometry without its fixed document identities.
Do not relax or mutate historical L10/L11 fixture validators to accept new data.
A distinct nonnormative snapshot schema must bind immutable vault ID, append
assigned document ID, revision, raw relative path bytes, source length/SHA,
physical manifest SHA and previous snapshot SHA. Content identity may deduplicate
physical objects but must never merge logical documents.

Initial admission: Linux local filesystem, at most 64 active documents,
256 bytes per file, regular files only, no symlinks or special files. An explicit
isolated synthetic source directory is required. Read with O_NOFOLLOW, bound the
read itself, and compare pre/post descriptor metadata. Rejected import must not
advance current snapshot. Missing paths are not implicit deletion. Rename and
revision require explicit document identity; no heuristic inference from content.

Serialize writers with flock; compare expected parent under the lock. Stage
immutable physical objects, independently restore and compare SHA/length, fsync
all files and object directories, install without overwriting existing content,
fsync parent directories. Write/fsync snapshot before atomic replace of CURRENT;
fsync its parent before returning durable success. Retain all prior snapshots
and referenced objects. Orphan staging after failure is permitted and never
visible as a published snapshot. No garbage collection in this slice.

Queries require an explicit externally pinned snapshot and exhaustive active
leaf validation, even for absent patterns. Coordinates include vault/snapshot,
doc ID/revision/byte offset. Enumeration cap is distinct from full count and
coverage. Any missing/corrupt required object or wrong pin is UNTRUSTED. Independent
restoration must not import producer helpers and must verify decoded bytes before
output publication. Empty sources have explicit authenticated representation.

Acceptance matrix: initial import; late early-sorting name; duplicate-content
new document; explicit rename; linked revision; empty and binary inputs;
unchanged older snapshots; missing/corrupt leaf; wrong root; capped enumeration;
writer conflict; process SIGKILL before/after each durable-publication boundary;
new-process restart yields complete old or complete new snapshot. Re-run old
snapshot queries/restores after each append and failed publication. SIGKILL tests
establish process-crash recovery, not physical power-loss durability.

Gate output must include exact implementation SHA, every command/exit, full
retained object/index/metadata/decoder bytes, elapsed time, CPU, RSS and temporary
disk scope. No 1GB, cross-host, storage saving, signed currentness or antirollback
acceptance follows from this small-file gate. Next step: implement the reusable
bounded physical-object builder and durable snapshot store, then execute this
matrix before calling the slice ACCEPTED.
