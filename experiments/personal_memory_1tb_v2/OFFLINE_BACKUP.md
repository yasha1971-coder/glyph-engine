# Consistent closed-store backup pilot

`packed_backup.create_backup(metadata, data, new_destination)` holds the existing
cooperative owner lock for the entire copy/readback. Dirty metadata, journals,
pending transactions, symlinks, overlapping paths and existing destinations are
rejected. Sources are read-only. Data and metadata are copied into one new root.
Every copied file is fsynced and read back by SHA-256. Directories are fsynced.
Only after success is BACKUP.json published; its SHA-256 is returned to the host.
Retain that pin independently. A sidecar stored with its own data is not an
independent trust anchor.

`verify_backup(destination, trusted_receipt_sha256)` checks the receipt pin,
exact file set, sizes and hashes. This proves copy identity, not that the original
catalogue was semantically correct, or that files were healthy before copying.
Restore requires the appropriate store reader and its normal end-to-end checks.

Scope: Linux, trusted local directories, a stopped application, cooperating writer.
Not protected against hostile replacement of parent directories, dishonest disks,
or a privileged process ignoring the lock. Default maximum copy size 256 MiB;
explicit maximum 1 GiB, 10,000 files. NOT a command to copy the user's 1 TB dataset.
No original files or incomplete destinations are automatically deleted.
Interrupted copies remain incomplete and are not resumable in this version.
An interrupted final publication may leave a receipt: require external pin and
full verification, not just file existence. Hardware power-loss is not tested.

Validation in assistant environment, synthetic data only:
Final targeted run 2026-09-14: 5 tests passed, 0 skipped, 0 failures,
0.036 seconds unittest time. This is test duration, not a backup throughput result.

* Copy/readback, rejected altered metadata and wrong pin.
* Busy store, pending recovery, symlink and insufficient budget rejection.
* Injected readback failure leaves no completion receipt and intact sources.
* Real experimental packed store: two versions with shared original data; copy
  verified, both original directories renamed out of the way, both versions
  restored byte-for-byte from backup with their history intact.

The real-store test is explicitly SKIPPED where packed_scale_store is unavailable;
the other tests exercise the copy protocol independently. Do not count that skip
as successful store restoration.

```sh
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_packed_backup.py -v
```

This restores from an intact paired backup, NOT from packs alone after loss of all
catalogue copies. Recovery manifests, large incremental backups, backup encryption,
off-device copies and GUI integration remain separate engineering gates. The
user's laptop data is untouched and the feature is not deployed there.
