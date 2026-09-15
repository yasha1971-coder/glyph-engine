# Codec frontier experiment V1

Goal: measure alternatives to the current always-run deflate9/bzip2-9/xz-9
preservation codec search. This experiment does not change app codecs or vaults.
Native Gear remains opt-in and is not in this independent compression path.

## Primary research checked 2026-09-14

* https://tukaani.org/xz/man/xz.1.html : presets use different dictionaries and
  match finders. -3 uses hc4, -6/-9 bt4. -6 is the default. Larger dictionaries
  are not universally useful and impose encoder AND decoder memory cost.
* https://github.com/facebook/zstd/releases : published v1.5.7 release notes
  describe small-block and --patch-from improvements. The local reference test
  uses installed libzstd 1.5.5, not an untested claim about the newest release.
* https://github.com/facebook/rocksdb/wiki/Compression : heavier compression can
  be placed at the bottom level rather than paid for every rewritten level.
  Useful architecture precedent, not a GLYPH benchmark result.
* https://www.usenix.org/conference/fast13/technical-sessions/presentation/harnik :
  compressibility filtering precedent. Our simple rule is NOT that algorithm.

## Experiments

legacy-best, bzip2-9, xz-3, xz-6, xz-9, sample-bz-xz6, and zstd-3/zstd-9 when system
libzstd exists. Zstd uses the stable C API via ctypes with external exact restore;
no installation, auto-download or system change. Missing zstd is explicit skip.

The sample rule was frozen before this three-repeat experiment: three disjoint 64KiB
windows (start/middle/end), each compressed with bzip2-9 and xz-6. Choose whole-file
xz-6 only if samples suggest >=10% relative benefit AND >=256KiB extrapolated
saving; otherwise bzip2-9. Inputs <192KiB use bzip2 without probing. Sample cost
is included in encode time. Misclassification is allowed to lose density, but must
be reported. No file names, hashes, prior reports or per-file tuning in the rule.
Raw fallback is selected whenever the selected compressor expands the input.

Silesia was already known from earlier GLYPH results when these candidates were
chosen. This is development-corpus evaluation, not a blind validation set.
The rule reads only the current bytes; it does not consult the earlier results.

All policies use the same two workers by default, three runs in rotating order.
Each object is written, fsynced, reread, decoded and compared byte-for-byte.
Timing includes source read/hash, probe, codec, temporary I/O, exact verification
and pool startup. It excludes persistent catalogue construction/directory commit;
this is NOT an end-to-end application ingest benchmark. Readback may be cached.
RSS is each worker's lifetime high water, NOT an aggregate concurrency measurement.
Bytes are sums of payloads, not archive sizes. Identical input files are NOT deduped
here; use this experiment to isolate codec choice. Precomp is not benchmarked, so
VIKA results are not a substitute for the prior 21.25% reversible-precomp result.

Inputs: explicitly selected folder, depth <=6, <=10,000 directory entries,
1..500 files <=64MiB each. Password basename тут все.txt and backslash variant
excluded, symlinks skipped. The depth limit deliberately bounds discovery; this
is not an inventory of every descendant. Changed input, wrong hash, bad decode and existing
output cause STOP. Files are never uploaded or changed. Report has content hashes
but no filenames or document bytes; user should still choose which reports to share.

Run without competing scale tests:

    python3 experiments/personal_memory_1tb_v2/codec_frontier.py --source /path/to/input --output /path/to/new-output

Silesia first, VIKA as an independent heterogeneous check. Do not tune on VIKA
and still call it a held-out test. Criterion proposal: <=5% larger payload than
legacy best, materially faster measured full pipeline; no lossless failures.
Select nondominated size/time alternatives rather than forcing one global winner.

## Measured decision

See CODEC_FRONTIER_RESULTS.md and the two checked-in result JSONs. The adaptive
policy passed the proposed <=5% payload cost criterion on Silesia, with a 4.17x
pipeline speedup over legacy-best at equal concurrency. It remains experimental
until the same frozen policy has been evaluated on the current VIKA files and
then through the actual ingestion/restore path. The additional synthetic case
shows why probing is not free: identical density to bzip2-only, with more time.

Architecture direction: optional fast initial preservation, exact content/block
identity independent of physical codec, and deferred dense recompression only
for retained unique content. Recompression must verify the new representation
and atomically publish its reference before reclaiming the old one. This is a
design choice supported by the measurements and the RocksDB precedent, not a
claim that asynchronous repacking, crash-safe codec migration, or zstd storage
has been implemented in the application. Existing archive formats remain usable.
