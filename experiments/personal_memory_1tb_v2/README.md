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
