# Scaling research / 2026-11-08

Primary sources consulted:
- https://github.com/IlyaGrebnov/libsais — induced-sort construction; throughput
  depends on memory architecture. Reuse vendored frozen code; do not upgrade it
  silently. Native input-sized RAM remains distinct from bounded verifier buffers.
- https://github.com/lh3/ropebwt3 — dynamic build vs static/mmap search representation,
  per-input incremental merging, sampled suffix arrays. DNA/reverse-complement
  semantics and published terabase compression figures do NOT transfer to our
  arbitrary-byte Vault. No adoption of its ABI, formats, data or runtime here.
- https://restic.readthedocs.io/en/latest/047_tuning_parameters.html and
  https://github.com/restic/restic/blob/master/doc/man/restic-check.1 — verify newly
  generated objects before publication; distinguish structural from full data
  checks. Our acceptance still requires full source replay, not sampled success.

Implemented opt-in GLYPH_BOUNDED_IMPORT_STREAM_V2. Reuses native builder/durable
snapshot writer, existing L9-S rank geometry, source materialization, independent
streaming source hash+exact scan and structural index verifier. Original V0/4K and
historical fixed-enwik8 profiles retain their limits and evidence. New scale_leaf.py
is an explicitly separate derived adapter; historical source pins are not silently
redefined. Max source=100000000 bytes, component=128MiB, 64 docs/64 snapshots.

Trust tier is EXPLICITLY different: independent_index_source_semantics=false.
Source length/SHA and full logical answer/coverage are independently verified;
header/rank/EF structural geometry is checked, not every rank prefix or suffix
value independently reconstructed at scale. No succinct proof or universal index
correctness claim. Producer authenticates original source once during startup,
then deletes it before retained-index queries. Verifier materializes/scans all data.

High-count policy is versioned: if any leaf count >256, coordinates are omitted
and enumeration=OMITTED_HIGH_COUNT, status=UNTRUSTED. No invented partial canonical
prefix is called complete. Exact count still includes every mandatory document.
4K V1 retains its full bounded internal enumeration and first-256 policy.

Local ladder: 1MiB, 8MiB, 100000000 bytes/file, mixed synthetic repeated 64KiB block
plus a unique marker. Two snapshots and two logical documents share one content
object. This is highly repetitive data, not representative heterogeneous/VIKA data.
No source archive is part of retained Vault. Preserved ladder-1 failure: 8MiB hit
legacy identity helper's 1MiB index-component cap. Added separate streaming hash
for the scaled builder with explicit 128MiB cap; old helper/default limits remain.

Run scale_ladder.py --reference /ABS/REF --decoder /ABS/DECODER --output /ABS/NEW
[--sizes comma-separated-bytes]. It checkpoints result.json after each stage and
stops on first failure. A completed 100000000 stage reaches this protocol ceiling,
NOT maximum hardware capacity. Counts/absence/restoration are checked on all stages.
scale_negatives.py rejects forged counts, false strong-semantics flag, omitted
coverage and missing/corrupt EF. CI additionally replays existing streaming boundary
and overlap tests, previous profiles and a separate runner on the 8MiB capsule.

Resource fields: elapsed stage wall; self/max-child RSS separately and cumulative
own/terminated-child CPU. These are not simultaneous process-tree memory or isolated
per-stage child high-water marks. Retained logical Vault bytes include decoder,
indexes and metadata; installed toolchain/code/cache, original sources, build SA/BWT,
source materializations and peak allocated temporary blocks are separate/excluded.
No industrial deployment, 1GB acceptance, hardware maximum or storage-ratio claim.
