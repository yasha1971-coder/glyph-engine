# Offline checkpoint guard after the 1 TB experiment

The supplied SCALE_CEILING.json reports 1,000,000,000,000 unique generated raw
bytes and two complete successful scans. It is not a GUI/compression benchmark.
Receipt SHA-256: e5d9e269872cf4cf21ab6bf376ef2b8bd2cdda9af3a8fb4fce100e4a850ff2b6.

Additional assistant-local synthetic experiments:
* Eight existing packed-store tests passed in 6.632 s, including real SIGKILL at
  three commit boundaries. This is not power-cut testing of the laptop disk.
* An offline copy of metadata and packs restored a two-version document exactly.
* Replacing only the copied metadata database with an older database from the
  SAME store opened successfully and exposed only the older version. Identity
  binding alone does not detect rollback or prove history completeness.

New `packed_checkpoint.checkpoint(metadata, data, expected=trusted_pin)` hashes
the closed SQLite database and data identity while holding the existing owner
lock. It rejects pending recovery/journal files, unavailable locks, direct file
symlinks and a mismatched expected pin. Three targeted tests passed in 0.007 s.
The module is independent of the experimental packed-store implementation.

Workflow: after successful close and scrub, explicitly capture a checkpoint and
keep its pin independently from the backup. Verify it before reopening an exact
frozen copy. A later legitimate transaction also changes the pin; this is NOT a
live-store anti-rollback implementation. Never auto-refresh a trusted expected pin
when verification fails. An attacker replacing both checkpoint and data defeats
an untrusted sidecar, so independently retained provenance is essential.

The guard does not parse or reconstruct a damaged SQLite catalogue, validate all
payloads, make backups, or replace fsync/hardware testing. No laptop files changed.
Loss of the catalogue still requires a consistent metadata backup. A persistent
version-manifest recovery design is needed before packs alone can reconstruct
document histories. No claim of production readiness is made.

References consulted 2026-09-14:
https://kopia.io/docs/advanced/architecture/
https://www.sqlite.org/howtocorrupt.html
https://restic.readthedocs.io/en/stable/045_working_with_repos.html

The useful industry principles are independently recoverable metadata, coherent
backup copies and a distinction between index checks and verified data reads.
These references do not certify GLYPH or establish a measured comparison.
