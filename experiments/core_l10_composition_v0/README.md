# CORE-L10 — immutable document composition and append research

Task [#28](https://github.com/yasha1971-coder/glyph-engine/issues/28), additive
Draft stacked on PR #27 at `e7f0e3b16381519c8b0269f6fea0ff11feb0cc80`.
Read [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md) first: Gate 0 was
completed and the unchanged historical checker/independent replay executed
before this adapter was written. Historical operator format is **not** RLB2/EF.

Verdict **VERIFIED_APPEND_RESEARCH** requires all three CI jobs, actual leaves,
both layouts, old/new snapshots, complete independent replay and all hostiles.
Pending CI is not success. This is a bounded research adapter, not Index
Forest, a shared ABI, production memory service or an authenticated current
world state. No main/frozen/CONTROL/OVH/user-data/deployment/merge action.

## Compatibility and actual execution

The exact historical identity preimages are reused for **logical**
`runtime_corpus_id` and `source_manifest_id`. The former includes dense doc
positions/lengths/raw source digests; the latter additionally includes canonical
relative path bytes. They are not P4 name-bearing proof identity or serialized
source-manifest-file SHA. Actual old checker confirms the first nine documents:

```
runtime_corpus_id = 4765ce3496058719db1436c56a33b325b16dc1c4cdfecd86664d0bdfb94dc950
source_manifest_id = a21e00fea40c4f7fb31b22c86400436acd90dba6e6efba855b4395ce7d162e09
```

Physical **CONTRACT_MISMATCH** remains explicit. Old operator blocks retain
SA/BWT/FM, source snapshots, manifests and complete markers, with an old
`runtime_index_id` / `composition_root_id`. New physical state uses one real
RLB2/RLR2/physical-EF index per nonempty document, separately bound to its doc
slot/path. New flat grouping manifests are called research blocks. They are
not fed into, labeled as or claimed compatible with old operator blocks.
No persistent combined FM/routing structure or Index Forest runtime is added.

New root format is exactly `GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0`.
`research_composition_root_id` hashes the distinct domain
`PROFILE || 00 || "ROOT" || 00` plus canonical root body excluding that field.
Full root JSON SHA256 is a separate caller pin. Grouping, document/leaf order,
physical commitments, code/decoder and parent identity are bound. This digest
is never aliased to the historical layout root or an operator index ID.

`fixture.py` is construction-only. It uses canonical small SA/BWT construction
and the original **frozen** Python RLB2 encoder/rank builder at
`c33cb493bc75f13ee4a904fabb190a29a196c94b`, plus existing physical EF builder
and sample verification. Construction SA/BWT files are removed before query
processes start. The temporary SA uses the existing EF builder input layout;
it is not represented as a retained canonical SA artifact or a mock backend.
No construction/source module is imported by query code.

`producer.py` uses unchanged L9-A/B `QueryState` / real RankV2 and EF extraction.
Every nonempty leaf source is reconstructed and SHA-checked, and actual FM
count/full locate/excerpts are used. Primitive L9 receipts remain byte-schema
V0, unchanged. Their result cap is never used as a full global allowance on
each block: complete counts are obtained from every leaf, then exactly one
canonical global prefix is selected. Full coverage and offset completeness are
distinct. Any incomplete enumeration yields UNTRUSTED/LIMITED, retaining count.

One empty document is an explicit, hashed **EMPTY_MANIFEST_ONLY** binding,
length zero/SHA256(empty), with no fake index or primitive receipt. It remains
in logical identities and later doc bases. The independent verifier requires
the binding and a real zero-length oracle object and hashes it itself. Missing
empty binding/source is not silently interpreted as an empty document.
This preserves old nonempty-query empty-set semantics; physical old indexing
of empty documents is not claimed compatible.

Byte-identical documents occupy different physical leaf directories and have
different binding hashes because doc IDs/paths are bound. Equal payload or
component digests do not merge logical slots. Canonical coordinates remain
`[global_doc_id, document_byte_offset]`; block identity is only provenance.
Querying separate document indexes structurally excludes concatenation traps.

## Snapshots and lineage

V0 contains exactly the original nine-document historical fixture: empty,
alpha/edge/shared/binary documents, two `dup` documents at IDs 5/6, tail/omega.
Layouts A0/B0 group the same sequence as **3+3+3 / 2+4+3**. They have equal
global IDs/answers and distinct physical research root IDs.

V1 appends three documents at IDs 9/10/11: `ONLY_IN_V1 shared`, a third `dup`,
and binary/spacing content. A1/B1 extend their old grouping with one new block.
Old source IDs/coordinates stay stable; global corpus/source/layout IDs change.
Old physical leaves, block/root files and **40 old V0 receipt files** are hashed
before append and compared byte-for-byte identities/sizes after append. The
old receipts are reused, not regenerated, for postappend V0 replay.

The independent verifier recomputes **full ordered old/new document and leaf
manifest prefixes**, old physical commitments and the initial block prefix.
It requires a separately supplied expected V0 root pin; a stored parent link
alone is insufficient. The producer captures V0 pins before append. CI carries
all four expected root pins via a separate job output, outside the index-only
capsule; the second job checks them before decoding or replay. These pins are
a research trust assumption, not author signatures or currentness witnesses.

V0 proofs remain scoped to V0. Extra unreferenced leaves in the immutable pool
do not invalidate an old root or become part of its declared coverage. A V1
receipt must cover every V1-committed leaf. This is not a freshness test that
claims the filesystem contains no newer uncommitted generation.

No Merkle consistency proof is implemented. Ordered-manifest prefix replay
under a trusted old pin is explicitly not RFC 9162 CT consistency. No anti-fork,
rollback, signature or independently controlled authority is supplied.

## Independent verifier and gates

`verifier.py` has standard-library imports only, no import of this composition
producer/helpers. It independently parses strict canonical JSON and rebuilds
logical IDs, physical group/leaf binding, exact coverage, generation and lineage.
It hashes each source oracle and scans every document independently, including
overlaps, counts and exact global prefixes/excerpts. It invokes the unchanged
**independent L9 small verifier** in separate processes for every nonempty
leaf receipt. That verifier imports no producer and independently checks the
RLB2/RLR2/EF source semantics through its bounded source-derived SA/BWT oracle.
Thus the layered proof executes real leaf evidence, not source-only mock leaves.

Oracles are ephemeral and outside retained query state. Unchanged native L8
materializer produces them from physical indexes after construction inputs are
deleted. The producer runs in a separate process while the oracle directory
is withheld. Independent CI re-materializes its own oracles after capsule
inventory and externally expected root-pin checks. No source oracles are uploaded.

20 query/policy cases cover shared/duplicates, NUL/FF and adjacency, missing,
case/spacing variants, cross-doc and cross-block-only traps, a query longer than
any document, and global caps 0/1/2. There are 40 V0 replays before append and
80 V0/V1 replays after append in the producer job; independent job replays all
80 again from its own source oracles. Source coordinate expectations include:

- `dup`: V0 `[[5,0],[6,0]]`; V1 adds `[10,0]`.
- `ONLY_IN_V1`: V0 PROVEN_EMPTY, V1 FOUND `[[9,0]]`.
- `shared`: V0 three hits, V1 five; global caps preserve full count.
- Physical `00 ff shared` and `dupdup` concatenation traps are absent in each
  document under both layouts; no cross-boundary filtering after counting.

38 named hostiles include missing block/leaf/empty binding, stale/duplicate
leaf, order changes, merged identical docs, removed empty doc, missing V1
coverage falsely claiming zero, forged generation, wrong old pin, changed old
leaf bytes, unacknowledged new leaf, cross-boundary fake hits, fabricated doc ID,
per-block cap, SHA/excerpt/query/root/count/policy/shape forgeries, old receipts
replayed against new version/layout, and re-pinned malformed EF checkpoint.
For the last test the entire binding/block/root hash graph and leaf receipt
pins are rebuilt around the malformed EF. Source identity and file geometry
remain valid; rejection must explicitly reach the independent leaf physical
oracle. The real producer must also reject it before an absent-query claim.

Full valid replay: exit 0, `VERIFIED_APPEND_RESEARCH`. Correct limited receipt:
exit 3, `UNTRUSTED / LIMITED_COMPLETENESS`, independently verified full count
and coverage. Any invalid/missing/partial evidence: exit 3/UNTRUSTED, never
PROVEN_EMPTY. CLI syntax errors use argparse exit 2. All needed calls actually
execute; there is no optional/mock backend or skipped required case.

Historical gate separately checks out
`81628f843dc8aa633c65594a66d2386bfc3cb5c2`, compiles its original operator
binaries and runs the **complete unchanged** checker: 25 EXACT obligations,
34 implemented mutations, and the original separate replay (two deterministic
positive runs, three rejected negatives). It remains research tier and is not
added to top-level VERIFY closure.

## Costs, bounds and reproduction

Initial profile: 9..64 docs, <=256 bytes/doc, generations 0/1, append >=3 docs,
safe canonical ASCII paths, <=256 returned coordinates, root <=64KiB, receipt
<=512KiB. The old L9 4096B cap and L9-S 100MB restrictions are unchanged. This
gate is not a 100MB multi-document, 64-bit/TB or arbitrary version-history test.

Query/replay covers **all** blocks/leaves: O(blocks + documents) fanout plus
source-byte scans. Layered small leaf verification additionally costs bounded
per-document quadratic SA/BWT reconstruction. It does not inherit a scalable
whole-composition proof from L9-S. Snapshot loading validates all dependency
hashes; stdout/result artifacts record its startup and per-query CPU/wall/RSS,
root/block/leaf hash/read work, leaf-verifier process/source/SHA work, every
root identity, and actual preserved/reused/new file bytes and receipts/code.
Startup timing is wall time. `cpu_seconds` is the measured Python process's
own CPU, not aggregated child CPU; replay separately records child peak RSS.
Native materialization/construction CPU and simultaneous process-tree peak
RAM are not isolated by this gate and no total-process CPU/RAM claim is made.
Bundle inventory counts a shared decoder once, distinct duplicate leaves twice.
Temporary snapshots/oracles/mutation copies and construction binaries are not
free. Reports separately inventory support sources/docs with SHA256, complete
research/reference checkout file lengths and unique Git stores (these overlap
named code and must not be blindly added to it). Per-snapshot private
component/decoder copy lengths and the largest native materialization copy and
candidate are derived from the actual physical file lengths. Scratch file
lengths are not sampled peak allocated disk blocks. Platform
Python/stdlib/compiler/git installations and full source checkouts are external
costs, not included in a total installed footprint or compression-ratio claim.

From a disposable checkout, with `$REFERENCE` at frozen C33, `$HISTORICAL` at
81628f8, and unused `$SCRATCH` output paths:

```bash
source /workspace/glyph-env/activate.sh
cmake -S "$HISTORICAL" -B "$HISTORICAL/build"
cmake --build "$HISTORICAL/build" --target build_sa_binary_v1 build_bwt_binary_v1 \
  build_fm_binary_v1 query_fm_binary_v1 query_fm_locate_binary_v1 -j2
python3 -I experiments/core_l10_composition_v0/historical_gate.py \
  --historical "$HISTORICAL" --output "$SCRATCH/historical"
g++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread \
  experiments/loc_text_ef_v0/native_run_stream_inverse_rlb2_ef.cpp -o "$SCRATCH/l8-decoder"
python3 -I experiments/core_l10_composition_v0/gate.py --phase produce \
  --reference "$REFERENCE" --decoder "$SCRATCH/l8-decoder" --output "$SCRATCH/produce"
# EXPECTED_PINS comes separately from the trusted producer, not the capsule.
python3 -I experiments/core_l10_composition_v0/gate.py --phase independent \
  --reference "$REFERENCE" --capsule "$SCRATCH/produce/capsule" \
  --expected-roots-json "$EXPECTED_PINS" --output "$SCRATCH/independent"
```

CI has historical / producer-and-append / independent-composed-replay jobs.
Every checkout pins exact PR head or immutable full SHA. Capsule <=4MiB,
index-only, retention 14 days; evidence retention 30 days. Final report/PR
records actual HEAD and CI conclusions rather than inferring success from code.

## FALSIFIERS / FAILED_PREDICTIONS and open boundaries

Gate 0 full historical checker/replay passed before coding. First local full
adapter run passed all 120 replays and 38 hostile cases; no semantically wrong
hit/absence was accepted. Expected-pin transport and explicit V0 receipt hash
preservation were then added and rerun successfully, including the separate
80-replay capsule run. A deliberately incorrect separately transported A0 pin
was rejected before copying/decoding the capsule. Additional support/snapshot/
Git-store accounting changes only the driver report and is rerun as well.
All later attempts/outcomes must be
reported honestly; pending CI is not relabeled PASS. The earlier L9 malformed
checkpoint falsifier remains preserved and is explicitly retested here.

Root authority/currentness, rollback/fork, Merkle consistency, portable O3/O4
production closure, arbitrary append/repartition/code upgrades, 64-bit/TB,
normative ABI, Index Forest and fixed SLA remain open. Old six packed-backend
SKIPs and missing bounded-memory RLB3X patch lineage remain open. Nothing here
promotes old physical Composition formats into a new common ABI or rewrites
historical checkpoints. Owner retains veto on promotion, merge and deployment.
