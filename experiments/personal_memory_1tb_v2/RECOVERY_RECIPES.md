# Recovery recipes: independent of the SQLite catalogue

## Decision, checked against primary sources on 2026-09-14

Kopia separates content blocks, indirect object descriptions, manifests and
indexes, and places a recovery index in each pack. Restic separates immutable
content-addressed repository files, snapshots, trees and pack indexes. Borg also
documents separate repository/manifest/archive structures. The useful common
principle is that a fast lookup database need not be the only description of
what belongs to a document version.

Sources:
https://kopia.io/docs/advanced/architecture/
https://restic.readthedocs.io/en/stable/100_references.html
https://borgbackup.readthedocs.io/en/stable/internals/data-structures.html

These are design references, not proof that GLYPH is faster, safer or the best
available system. No new encryption design, distributed consensus, or format
migration is warranted by the 1 TB raw-storage experiment alone.

## Implemented bounded experiment

`recovery_recipes.export(store)` exports version IDs, names, recorded creation
timestamps, byte counts, hashes, profiles and ordered block locators while the
caller owns Store's existing lock. Each recipe is verified with an independent
reader before the export returns capsule bytes and their SHA-256 pin.

`load(raw, trusted_pin)` reads that independent catalogue. `restore(raw,
trusted_pin, data_root, version_id)` needs the capsule and original packs, but
NO SQLite database. It checks the data-store identity, frame headers, block
hashes, total size and whole-version hash before returning complete bytes.
Names are catalogue labels, never filesystem output paths. The module performs
no writes or source deletion. Keep capsule bytes durable and the trusted pin
independently; return from export alone is not durable publication.

Local synthetic tests 2026-09-14: 4 passed, 0 skipped, 0 failures in 0.017 s.
The integration test created two real GLYPH versions sharing original blocks,
exported descriptions, closed the store and renamed the original metadata
directory out of reach. Both versions were restored exactly without opening
SQLite. Other tests cover missing/corrupt packs, wrong identity/pin, duplicate
version IDs, invalid locators and unknown versions.

```sh
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_recovery_recipes.py -v
```

Where experimental packed_scale_store is absent, its integration test is
explicitly skipped; standalone reader tests still run. Test duration is not a
throughput measurement. Capsule syntax and decoder are pilot code, not audited
hostile-input parsers. Trusted local filesystem/process and externally trusted
capsule pin are assumed; hostile concurrent path replacement is out of scope.

Limits: capsule <=8 MiB, <=1000 versions, <=10000 blocks per version, <=64 MiB
per restored version, <=128 MiB total exported version bytes. Recipes repeat
locators across versions; persistent shared recipe trees remain future work.
This is NOT a 1 TB export. No payload copying/compression changes were made.

## Still required before deployment

This is an explicit closed-state export, not an automatic per-commit recovery
manifest. Versions added after export are NOT covered. Lost capsules cannot be
recreated from packs alone with names/order/history guaranteed. Lost payloads
still need a separate backup. Index rebuilding, deletion/tombstone semantics,
authenticated complete commit history, capsule encryption, bounded streaming
export/restore and fault-tested atomic publication remain unimplemented.

The next gate is commit semantics: publish a recoverable description without
making an uncommitted version look committed after interruption. Do not silently
enable this for the user's existing laptop vault. No original laptop data changed.
