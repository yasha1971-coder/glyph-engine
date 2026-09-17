# Offline personal-memory data backup

Implemented 2026-09-17 for incremental_memory.Memory, the backend used by the
actual personal browser. This is not the separate experimental packed backend.

Design reference: restic separates repository checking from restoring snapshots.
https://restic.readthedocs.io/en/stable/045_working_with_repos.html
https://restic.readthedocs.io/en/stable/050_restore.html
This implementation does not use restic's format and does not claim equivalence.

## Included and excluded

Copies both immutable base archive and entire overlay (CURRENT, snapshots,
objects, recipes and chunks). Source UI must be closed; both browser and writer
locks are held throughout copying and validation. Pending deletion is rejected,
not silently recovered or copied. No original files are modified or deleted.

Code, Python/runtime dependencies and Precomp executable are NOT included.
When required, a trusted compatible Precomp executable must be supplied separately;
its hash is retained. No executable from a backup is automatically run.
This is unencrypted data backup: filenames and contents may be sensitive.
It is not an off-device backup unless the chosen destination actually is off-device.
The immutable base can retain documents deleted from the active overlay.

Files are copied in 1 MiB buffers, fsynced and read back for SHA-256 comparison.
After copying, the new Memory is opened with the pinned archive receipt and all
unique reachable history items are decoded and byte-hash checked. Only after
success is PERSONAL_BACKUP.json published. Keep the returned receipt SHA-256
independently; the hash inside an untrusted backup is not its own trust anchor.

Restore verifies that trusted receipt, file membership, sizes and hashes; copies
to a NEW directory; opens and verifies the reconstructed memory; then publishes
RESTORE_COMPLETE.json. Failed directories remain for diagnosis without a success
marker. They are not safe to activate. Never replace the active application data
automatically. Backup/restore are explicit CLI operations; GUI integration pending.

## Bounds and threat model

Default copy and decoded-history budget: 1 GiB each; explicit --max-bytes may
increase to 64 GiB. Max 100,000 copied files; history limits inherited from the
application (1,000 snapshots / 64 MiB expanded metadata). These are bounded pilot
limits, not support for backing up the 1 TB scale experiment.

Trusted local filesystem and cooperating writers assumed. No protection against
a malicious same-user process replacing ancestor directories. Symlink entries,
special files, overlapping destinations, active writers and changed copy sources
are rejected. Semantic checking uses existing decoders and their limits; not
certified for hostile inputs. No hard process deadline, encryption, automatic
retention, physical power-cut validation or cross-platform restore guarantee.

## CLI

```sh
python3 personal_backup.py create --memory MEMORY --archive ARCHIVE --archive-sha256 TRUSTED_ARCHIVE_HASH --backup NEW_BACKUP_DIR
python3 personal_backup.py verify --backup BACKUP_DIR --backup-sha256 INDEPENDENT_BACKUP_HASH
python3 personal_backup.py restore --backup BACKUP_DIR --backup-sha256 INDEPENDENT_BACKUP_HASH --destination NEW_RESTORE_DIR
```

For Precomp archives add --precompressor TRUSTED_BINARY and, when creating,
--precompressor-sha256 TRUSTED_BINARY_HASH. These examples are developer interfaces,
not a request for the end user to perform additional experiments.

## Automated evidence

Five targeted tests passed in the development environment. The recovery test
moves BOTH original memory and archive paths out of reach, restores from backup,
compares old and new file bytes, and adds another version to recovered memory.
Other tests reject locked UI/writer, corruption, existing output, symlinks,
pending deletion and too-small budgets; simulated disk-full copy failure leaves
no completion receipt and preserves the source CURRENT.

Synthetic fixtures only. No real laptop files were read and no live laptop backup
was made. This component is necessary but insufficient for industrial readiness.
