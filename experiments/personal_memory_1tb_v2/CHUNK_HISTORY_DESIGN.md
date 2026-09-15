# Per-file history and economical version storage — 2026-09-13

This note supersedes the whole-file-only writer and global-version-menu status
in the preceding incremental/browser pilot notes. Original storage remains
readable; new snapshots use GLYPH_INCREMENTAL_MEMORY_PILOT_V2, optionally wrapped
in GLYPH_SNAPSHOT_PACK_V1 (bounded DEFLATE/base64 within JSON). Old executable
versions cannot read the new format: do not downgrade code against a new CURRENT.
No migration, deletion or recompression of the existing base archive is performed.

## Research actually read

1. Restic design: https://restic.readthedocs.io/en/stable/100_references.html
   Immutable objects/packs, pinned identities and content-defined Rabin chunking.
2. Borg structures: https://borgbackup.readthedocs.io/en/stable/internals/data-structures.html
   Fixed and buzhash boundaries; insertion resilience; chunk-count/RAM tradeoffs.
3. Kopia compression: https://kopia.io/docs/advanced/compression/
   Chunk, identify duplicates, compress new content. This is a design precedent,
   not a performance comparison with our code.
4. FastCDC paper abstract (USENIX ATC 2016):
   https://www.usenix.org/conference/atc16/technical-sessions/presentation/xia
   Gear-based CDC with skipped minimum and normalization. The implementation here
   is an original simple Gear reference, NOT FastCDC, not its full optimizations,
   not evidence of its reported speed.
5. Google open-vcdiff README search result:
   https://github.com/google/open-vcdiff
   VCDIFF binary delta alternative. No full implementation audit or benchmark.

No comparative production benchmark or representative personal-data survey was
performed. No universal optimality or improvement over Borg/restic/Kopia claimed.

## Decision and implementation

Use content-defined chunks to tolerate byte insertions that disturb fixed
alignment. Dedup identity is SHA-256, not the rolling boundary hash. Choose between
whole raw/DEFLATE and chunk recipe by newly needed payload plus recipe bytes;
retaining the existing history is a common cost. Small/highly compressible files
can stay whole. Existing identical files remain whole-file deduplicated.

Chunker parameters are fixed for this pilot: minimum 4096, mask target 16384,
maximum 65536 bytes (actual average is corpus-dependent). Versioned deterministic
Gear table derives from SHA-256 of GLYPH-GEAR-V1 plus byte value. Hash boundaries
are not privacy protection. No encryption is implemented by this layer.

New chunk payloads are independently raw or DEFLATE-6, content-addressed and
verified on reuse/read. Ordered immutable recipes bind chunk size/hash and final
file hash. A recipe may instead reference verified ranges of ONE legacy whole
object. This lazily reuses original archived bytes when a file is first edited;
it does not duplicate or proactively repartition the old archive. Range sources
must be base/raw/deflate, never another recipe. Subsequent recipes reuse these
same direct references. Thus there is no accumulating decode delta chain.
Restoring a range of a legacy Precomp file still decodes that whole legacy file.
A changed chunk is stored in full; storing literally only modified bytes is not
promised. Re-encoded compressed media may have little reusable content.

Default UI shows current paths only. Each has History and New version actions.
History lists only content changes of that path with UTC dates for new snapshots;
old timestamps are explicitly unknown. Selecting New version binds the path in
the form, independent of the uploaded filename. Default duplicate uploads open
the existing file history without adding another path, after byte verification.
A new upload cannot silently overwrite an existing name: use New version.
Existing duplicate path records are not automatically removed.

CURRENT listing reads the current snapshot only. Per-file history still traverses
bounded snapshot history (<=1000); full file maps still exist but are compressed.
This is not a persistent Merkle catalog or an industrial-scale index yet.

## Evidence and limitations

CHUNK_HISTORY_TEST_OUTPUT.txt records the complete synthetic suite. New tests
cover insertion reuse, no recursive ranges, old snapshot compatibility, packed
snapshot rejection, corruption, fallback compression, same-path version upload,
duplicate handling and original UI contracts. Automated tests use HTTP and real
worker processes, not graphical browser automation. Laptop acceptance pending.

Reproduce growth test from repository root:

    PYTHONDONTWRITEBYTECODE=1 python3 experiments/personal_memory_1tb_v2/measure_version_growth.py

VERSION_GROWTH_RESULT.json: seed 1984, initially 524288 random bytes, 100 iterations
of prefix insertion plus one changed byte, 416 catalog paths (415 synthetic
placeholders/base entries plus the edited file). All 101 versions restored and
SHA-256 checked. Whole-file raw/DEFLATE payload baseline: 52958138 bytes.
Final overlay INCLUDING all stored snapshots, recipes and payloads: 5405049 bytes.
Breakdown: objects 524288; chunks 3994351; recipes 650486; snapshots 235924.
Base archive excluded from both. Disk allocation, decoder distribution, temporary
space and CURRENT are not included; the benchmark exercises storage API, not UI.

Earlier parameter runs, before snapshot compression, are retained in
VERSION_GROWTH_16K.json (12120235), VERSION_GROWTH_32K.json (19881501), and
VERSION_GROWTH_64K.json (27501366). The 64K run preceded addition of parameter fields;
it used min=16384, target=65536, max=262144. These are exploratory comparisons on
ONE synthetic edit pattern, not a universal tuning result. Runtime values are
observations, not controlled cross-parameter speed comparisons.

The existing VIKA result remains ~21.25% for the original archive. The new result
must NOT be presented as VIKA compression or average personal-media compression.
Upload <=8 MiB, reference legacy reads <=64 MiB, 10000 file paths, no encryption,
no LLM, no permission/empty-directory backup; base archive remains necessary.
Repeated reads and many small filesystem objects can be costly. Next industrial
steps: independently test other edit/media patterns; packed chunk storage and
persistent authenticated metadata indexing; recoverable commit journal and
backup/export; encryption/key recovery; only then broader scale qualification.
