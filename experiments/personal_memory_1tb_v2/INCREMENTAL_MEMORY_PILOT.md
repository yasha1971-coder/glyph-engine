# Incremental Personal Memory: private storage pilot

Status: CLI implementation and synthetic tests. Not yet connected to the browser;
not installed or tested on the user's laptop. Not a production release.

The user confirmed a working browser download and readable restored file on
2026-09-13 after browser commit 54e4aad8266684dd900311f3960d0455eaeadc00.
That is user-reported real-device acceptance of one download, not a new full
corpus verification. The earlier reported 21.245% saving belongs to the existing
precompression archive; no space or speed claim for this incremental layer.

## Semantics

- Reference the caller-pinned existing archive without copying/recompressing it.
- A snapshot stores the full path-to-content map and its parent hash. It is
  immutable and addressed by SHA-256. Caller explicitly retains/selects a pin.
- Add is an overlay of a separate source directory, relative to that directory.
  Missing paths are retained; it does not mirror source deletions.
- A changed path creates a new version. The old snapshot stays readable.
- Equal bytes share a whole-file object, including across ancestor snapshots.
- An identical addition returns the existing snapshot pin without publication.
- Writers use a nonblocking process lock. Object and snapshot publication use
  fsync and an exclusive hard-link, preserving existing names. A failed addition
  can leave unreachable objects; it does not publish a partial snapshot.
- No mutable HEAD is maintained: adding to an old pin intentionally creates a
  branch. A future UI must remember the current pin and display conflicts.
- Restore verifies selected bytes before opening a new output file exclusively.

## Limits and unresolved work

Linux/WSL, trusted local owner, base archive must stay available. This is not a
self-contained backup, encryption, authenticity signature, Index Forest or LLM
integration. SHA pins require a separately trusted record; directory access alone
cannot establish the intended latest version or detect rollback.

New input <= 8 MiB/file, <= 10000 entries/add and <= 10000 paths/snapshot;
metadata <= 16 MiB/snapshot and <= 1000 ancestor snapshots. Full metadata is
loaded, with bounded history count; no industrial startup/RAM claim. Existing
Precomp backend limits apply; this CLI does not add the browser's child-process
resource isolation. Only use existing trusted archives/binaries. New payloads
choose raw or DEFLATE-6; there is no repeated codec search or 21.25% guarantee
for additions. Empty directories and filesystem metadata are not captured.
Files are read stably one at a time; there is no atomic filesystem-wide snapshot.
Concurrent adversarial filesystem mutation and physical power-loss behavior have
not been qualified. No garbage collection; failed writes may consume space.

The existing V1 segment writer was inspected, but is not reused as a writer:
it builds a full BWT/RLB3X representation and would change the storage-cost basis.
This prototype reuses the existing pinned compressed reader instead.

## Commands (developer interface)

Run from the repository root. Supply the SAME base/archive pin each time.
`--snapshot` is the full pin printed by init/add, not a manually guessed value.

```bash
python3 experiments/personal_memory_1tb_v2/incremental_memory.py init \
  --memory /absolute/new-memory --archive /absolute/existing-archive \
  --archive-sha256 BASE_RECEIPT_SHA256
python3 experiments/personal_memory_1tb_v2/incremental_memory.py add \
  --memory /absolute/new-memory --archive /absolute/existing-archive \
  --archive-sha256 BASE_RECEIPT_SHA256 --snapshot PREVIOUS_SNAPSHOT_SHA256 \
  --source /absolute/separate-incoming-folder
```

For Precomp objects also supply `--precompressor /absolute/pinned/binary` and
`--precompressor-sha256 BINARY_SHA256`. `list --snapshot PIN` returns the map;
`restore --snapshot PIN --path RELATIVE_PATH --output NEW_FILE` restores one file.
These placeholders are documentation, not a laptop command block.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/personal_memory_1tb_v2/tests -p test_incremental_memory.py -v
```

Eight synthetic tests: versions/dedup/revert and unchanged base; retained absent
files; snapshot corruption; object corruption and rejected reuse; injected failure
before snapshot publication and retry; concurrent writer; symlink/size rejection;
existing output preservation and overlapping source rejection. The injected
exception test does not simulate a real power loss. Full suite output is recorded
in INCREMENTAL_MEMORY_TEST_OUTPUT.txt.

Next: integrate selection of the latest snapshot, history and explicit additions
into a separate UI pilot, then test with a small user-selected incoming folder.
Do not re-inventory D or recompress the finished source corpus.
