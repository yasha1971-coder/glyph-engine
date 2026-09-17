# Bounded retries with stable operation identifiers

`idempotent_commit.commit_once(store, recovery_root, operation_id, name, chunks)`
is an opt-in adapter. The host must durably retain and reuse the SAME operation ID
and request on retry. A changed ID is a new operation, even with identical bytes.
A conflicting name, profile, compression choice or chunk sequence under an old
ID fails. The input fingerprint binds block sizes/hashes and those write options.

The durable retry_operations_v1 table lives in the same SQLite database. A
connection-local AFTER INSERT trigger binds the intent to the new version INSIDE
Store's existing transaction. Rollback undoes the binding; committed writes retain
it even if the process dies before publishing recovery HEAD or returning a reply.
Retry republishes the committed recovery capsule and returns the original ID.
No separate post-commit receipt is relied on to prevent duplicate creation.

SQLite trigger behavior checked against the official reference on 2026-09-15:
https://www.sqlite.org/lang_createtrigger.html
The TEMP trigger explicitly targets main.versions and is removed after the call.
Exclusive Store ownership and no reentrant/bypass writes on this connection are
required. The existing writer/default UI and 1 TB runner are not changed.

Validation: 2 test methods passed, 0 skipped, 0 failures, 0.268 seconds on the
assistant's synthetic real-store fixtures. Six subprocess SIGKILL cases cover
after intent, pack publication, SQLite commit, Store return, capsule publication
and HEAD publication. Each retries twice, verifies exactly one committed version
and one operation binding, and independently restores the expected exact bytes.
The other test checks replay, conflicting requests, different IDs producing two
versions, and shared storage of identical bytes.

```sh
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_idempotent_commit.py -v
```

The tests explicitly skip where experimental packed_scale_store is unavailable.
This is process-crash evidence, not physical power-loss testing or certification.

Limits: request buffered in memory, <=64 MiB and 10000 blocks. Existing bounded
full-catalogue recovery export limits still apply. Failed publication can leave a
committed operation without an acknowledged response; retry with the same ID.
If export limits are exceeded, retry cannot solve that capacity problem.
Intent rows are retained indefinitely in this pilot; no garbage collection.

Known boundary: recipe capsules do NOT yet include retry operation mappings.
Loss/rollback of the SQLite database loses the replay ledger. Do not reuse old
operation IDs against a reconstructed catalogue and claim duplicate suppression.
The next recovery-format decision must preserve operation identity as well as
document versions. No GUI integration, remote/multi-writer support or automatic
operation-ID lifecycle is claimed. No laptop data was accessed or modified.
