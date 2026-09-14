# Opt-in recoverable commit wrapper

`recoverable_commit.commit_version(store, recovery_root, name, chunks)` writes
through the existing Store, independently verifies/exports the bounded catalogue,
fsyncs an immutable capsule, and atomically replaces/fsyncs its HEAD. Only then
does it return a version ID and recovery pin. Caller retains the pin independently.
Store's lifetime owner lock is required; no separate concurrent writer is allowed.

The default Store and laptop format are unchanged. This wrapper is an explicit
experiment, not enabled in the GUI or the 1 TB runner. Existing recovery_recipes
limits apply (1000 versions, 128 MiB total version bytes, 64 MiB per version).
It reexports/reverifies the whole bounded catalogue on every commit; this is a
correctness prototype, NOT a scalable encoding path.

Failure after database commit but before successful return is ambiguous: the
new version may already exist. Call `publish(store, recovery_root)` to reconcile
the committed catalogue; do NOT blindly repeat the original add. This operation
does not create another version. The caller must separately determine whether
its intended write is present. Automatic idempotency keys are not implemented.

The published HEAD always addresses a completely saved capsule. Before HEAD
publication, old acknowledged versions remain described by the previous HEAD.
After publication, new versions are described as well. This does not promise
that a client knows whether a response was delivered. If SQLite is lost inside
the pre-publication gap, only the previously published versions are recoverable.
No unacknowledged-version recovery guarantee is made.

Interrupted temporary/orphan capsules are retained and ignored by HEAD reads.
A failure during the very first baseline publication can leave a nonempty root
without HEAD: this is explicitly rejected for inspection, not guessed/reused.
No garbage collection, capsule deletion, external rollback protection without
the separately retained pin, encryption or distributed consensus is added.

Validation on synthetic real stores in assistant environment (2026-09-14):
2 unittest methods passed, 0 skipped, 0 failures, 0.312 s. One method contains
four subprocess SIGKILL cases: after Store commit, after capsule fsync, after
capsule publication, after HEAD publication. Each checks old-version restoration,
exact expected published catalogue size, reconciliation without duplicate
versions, and exact new-version restoration. The other checks successful returned
pins and rejection of another store before writing any version.

```sh
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_recoverable_commit.py -v
```

Tests explicitly skip when the separately experimental packed backend is absent.
This is process-crash evidence, not a physical power-loss guarantee. Large-scale
shared recipe trees and atomic/idempotent application integration remain future
gates. User laptop datasets were not accessed or modified.
