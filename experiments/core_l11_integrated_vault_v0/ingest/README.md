# Bounded file ingest: research gate

Candidate: strict local Linux profile, <=64 snapshots in a parent chain,
<=64 documents/snapshot, <=256 bytes/file, one-level raw relative filenames.
Caller-selected sources only; gate creates its own synthetic files.
Uses existing frozen RLB2/RankV2/physical EF; historical code is unchanged.
Not a production filesystem backup, encryption layer or 1GB acceptance.

## Reproduce

Use Python >=3.10, Git, g++ and local filesystem fsync/flock/atomic rename.
Create an unchanged reference checkout at
c33cb493bc75f13ee4a904fabb190a29a196c94b. Compile:

```
g++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread experiments/loc_text_ef_v0/native_run_stream_inverse_rlb2_ef.cpp -o /absolute/new/decoder
python3 -I experiments/core_l11_integrated_vault_v0/ingest/run_gate.py --reference /absolute/reference --decoder /absolute/new/decoder --output /absolute/new/evidence
```

Output must not already exist. Gate emits result.json, independent.log,
writers.json, pins.json and an index-only capsule. Source files are ephemeral
synthetic data. Capsule retains original and newer snapshots but no original
raw file archive. Empty contents have explicit empty.json objects so artifact
transport cannot silently drop an empty logical document.

Independent replay uses only the capsule, separately supplied expected pins
and an unchanged reference checkout:

```
python3 -I experiments/core_l11_integrated_vault_v0/ingest/replay_capsule.py --capsule /absolute/evidence/capsule --pins-json '<exact pins.json transported separately>' --reference /absolute/reference --output /absolute/new/replay
```

If artifact transfer strips executable mode, restore only capsule/vault/decoder
mode to 0700. Its bytes are checked against the pinned snapshot and leaf roots
before native materialization. Existing workflow has separate producer and
independent replay jobs; the latter receives no construction source directory.

## Contract and boundaries

Stable append-assigned doc IDs; explicit doc_id updates increment revision.
Duplicate-content documents retain separate coordinates and share physical
objects. Missing source paths do not imply deletion. All old snapshots remain.
Schema and lineage reject numeric/bool aliasing, unsafe paths, deleted slots,
identity changes and revision jumps. Validation runs independently on each side.

Writer lock + expected-parent CAS; fsync object files/directories before snapshot,
fsync snapshot before CURRENT replace, fsync vault directory before success.
Owner-cooperative local storage is assumed: not protection against hostile
ancestor-directory replacement, disk firmware lying about flushes, source mutation
that escapes descriptor metadata checks, rollback or externally authenticated
currentness. Per-file reads are not a simultaneous filesystem snapshot.

Independent verifier imports stdlib only, restores every current document through
existing L8, replays existing independent L9 physical verifier and computes exact
logical coordinates/count/absence from recovered bytes. Full scans and quadratic
small-source physical oracle remain; this is not compact negative verification.
Limited enumeration remains UNTRUSTED even when independently confirmed correct.

Five deterministic SIGKILL boundaries exercise process crashes, not kernel crash,
power-loss, torn write or every possible syscall schedule. Two concurrent writers
must yield one success and one stale-parent rejection. Four source-admission
negatives, fourteen claim/schema/missing-leaf negatives and corrupted EF on absent
query must reject. These counts describe this gate, not exhaustive verification.

Costs in result.json include elapsed gate time, own and terminated-child CPU,
separate RSS high-water marks and retained logical Vault bytes including decoder.
Compiler, installed toolchain/Python/stdlib/Git, allocated blocks and simultaneous
process-tree peak are excluded. No whole-package compression ratio follows.

## Failed observations retained

Initial November 7 gate exposed retry of an already-written unpublished snapshot;
fixed by exact-byte reuse and re-fsync. November 8 source-admission test initially
expected the *old* renamed path z.bin to collide: that expectation was wrong,
since path reuse for a new doc is legal. Corrected negative to active a.bin.
The first local November 8 attempt lacked the old /tmp decoder; explicit decoder
was compiled anew. No failure is counted as a PASS or silently waived.

Next beyond this gate: broaden real-file size/admission and replace tiny quadratic
construction with a bounded-memory builder, then heterogeneous larger full-loop
and cross-host tests with full retained/dependency/transient cost accounting.
