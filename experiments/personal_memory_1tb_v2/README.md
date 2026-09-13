# GLYPH 1 TB Corpus Truth Gate V1

First successor-line tool after frozen GLYPH V1. It inventories source metadata
without reading file content, following symlinks or writing below the source root.

The SQLite state lives outside the source and makes directory traversal restartable.
The aggregate receipt records disk and mount facts, file-class totals, errors and
completion state. It deliberately does **not** claim compression, deduplication,
content identity or 1 TB acceptance.

```bash
python3 corpus_truth_gate.py \
  --source /mnt/d/GLYPH_V1_PORTABLE \
  --state "$HOME/GlyphPilot/GLYPH-V2-runs/disk-d-inventory-v1"
```

Exit code `0` means all reachable directories were processed. Exit code `75` means
the bounded run stopped before completion and the same command may resume it.


## Restartable content truth gate

`chunk_truth_manifest.py` consumes a completed metadata inventory and hashes
regular files in fixed 8 MiB chunks. Each checkpoint is committed independently,
so an interrupted multi-hundred-gigabyte file resumes inside that file. Source
files are read-only and payload bytes are not copied into the state directory.

Fixed chunks are the first measured baseline: SHA-256 runs in optimized native
code and proves aligned duplicate bytes. Content-defined chunking is a later
comparative experiment for shifted-content deduplication.

Exit code `75` means safely resumable incomplete work, `2` means fail-closed
source/state inconsistency, and `0` means a complete content manifest.

### One writer per chunk state

On Linux/WSL local filesystems the CLI holds an advisory `flock` on
`.chunk-truth.lock` from before opening the database through receipt publication
and database close. A competing invocation exits `75` with `state is busy` on
stderr, without opening the chunk database or publishing a receipt. It emits no
completion JSON; any existing receipt belongs to the previous completed run.

The kernel releases the lock on normal exit or process death. Keep the lock file
in place, including after a crash; its existence does not indicate a busy state.
Do not delete/replace the state directory or lock inode while a process holds it.
All writers must use this locking protocol. Older CLI versions, external SQLite
writers, hostile local changes and unverified network-filesystem lock semantics
remain outside this guarantee. No native Windows support is introduced.

## Fixed-chunk dedup baseline

After a chunk-truth run is complete, measure only the reuse that its fixed,
file-aligned SHA-256 chunks actually prove:

```bash
python3 experiments/personal_memory_1tb_v2/dedup_baseline.py \
  --state /path/to/chunk-truth-state
```

The command acquires the same cooperating-writer lock, verifies the canonical
chunk receipt and checksum, validates every database chain, counter, content
root and manifest root, then writes `GLYPH_DEDUP_BASELINE_V1.json` plus its
SHA-256 checksum inside the chunk state. It does not re-read source files, create
a payload store, compress data or claim shifted-content/delta savings. Identical
output state produces an identical baseline receipt.

## Verified whole-file hybrid archive pilot

`verified_hybrid_archive.py` is the first V2 step that creates compressed payload
objects and independently restores them. It content-addresses complete files by
SHA-256, stores exact duplicates once, and exhaustively selects the smallest of
raw, DEFLATE-9, bzip2-9 and XZ-9 for each unique object.

```bash
python3 experiments/personal_memory_1tb_v2/verified_hybrid_archive.py build \
  --inventory-state /path/to/completed-inventory \
  --output /new/archive/path \
  --required-saving-percent 30

python3 experiments/personal_memory_1tb_v2/verified_hybrid_archive.py restore \
  --archive /new/archive/path \
  --destination /new/restore/path
```

The reported storage ratio includes every object payload, the canonical manifest
and its checksum sidecar. The restore path verifies stored-object hashes,
decompresses through an independent command path, verifies every restored file
SHA-256 and reproduces the source manifest root. Source content and relative
paths are preserved; timestamps and other filesystem metadata are not yet part
of this pilot. It is whole-file routing only, not CDC, delta, cross-file
compression or a compressed self-index.

## Verified reversible-precompression pilot

`verified_reversible_precompression_archive.py` is a separate experimental
router for already-compressed JPEG, PNG, PDF and ZIP objects. It adds reversible
precompression candidates to the V1 whole-file codecs. A candidate is eligible
only after an immediate exact-byte restore; unsupported, failed or mismatching
candidates fall back to the ordinary lossless router and are counted in the
receipt.

The external executable is not bundled or trusted. Build requires its expected
SHA-256, and restore checks the receipt-bound executable hash again. The first
reference pilot pins upstream Precomp v0.4.7 source commit
`31b693d843e378e7d30190736f95c095769868b0`. The upstream release's Linux x86-64
binary observed by the pilot has SHA-256
`6a5289ba81ea658e8e7984bf32791635f53b8e7cee4aafc26b853c7cd5b018f8`.
The containing upstream `precomp.zip` release asset observed by the pilot has
SHA-256 `fcd308310135cdc1ae25b45288bcc5201f9cabf5a9e707cfb33b0f99c548de0b`.
Verify the downloaded release independently; do not treat these identifiers as
a supply-chain signature.

```bash
python3 experiments/personal_memory_1tb_v2/verified_reversible_precompression_archive.py build \
  --inventory-state /path/to/completed-inventory \
  --output /new/archive/path \
  --required-saving-percent 30 \
  --precomp /path/to/pinned/precomp \
  --precomp-sha256 6a5289ba81ea658e8e7984bf32791635f53b8e7cee4aafc26b853c7cd5b018f8

python3 experiments/personal_memory_1tb_v2/verified_reversible_precompression_archive.py restore \
  --archive /new/archive/path \
  --destination /new/restore/path \
  --precomp /path/to/pinned/precomp
```

This is a measurement probe, not a production Personal Memory dependency. The
upstream Linux release labels itself a development/test binary. Every selected
object is still verified locally, but future production integration requires a
reviewed, reproducible source build, hostile fixtures and explicit licensing and
supply-chain review.
