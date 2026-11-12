# Packed lifecycle gate — 2026-11-09

Bounded continuation of existing Issue28 / Draft PR29, not a new format or engine.
Start/lease: `4caec92838e35a388a115a452cb1cd15cb8983c4`, accepted tree
`15e1c8b4e8a1c7225b50bfd7288a85757ff51bf7`. Historical research run42588953212
has seven successful jobs; regular/append-identity runs also succeed. Those
results belong to that checkpoint, not this continuation. No other checkout,
main/frozen/CONTROL/tag/release/OVH/private input or canonical memory is modified.

## Baseline and corrections to prior evidence

Unmodified baseline packed_gate and replay_packed both exit0 locally, counts
0/1/59920, four negatives. Actual complete file lengths are **1173676B** versus
**2097171B** unique source versions (CI's different decoder build is1185468B).
Inspection additionally finds `staging/tmp*/check-859eeee...` **1048595B**, SHA256
`859eeee116016a7f71b51435e71efa283dc2b1df34af39e6549fde258c216bb1`, and staged
CURRENT65B after its before-current SIGKILL. This is a restored raw-source copy,
not an original pathname, but it contradicts the old stronger index-only/no-source
transfer statement. It was already counted in the old retained total. The old
baseline is preserved untouched; no retroactive correction of its artifacts.

A controlled counterexample with a revision, two current unique contents, and
query cases only at CURRENT deletes the entire ancestor-only object. **Old
replay still exits0 / CLEAN_PACKED_CAPSULE_PASS.** Its budget selected only query
snapshots and never restored that obsolete version. First attempted counterexample
instead failed on budget (exit1): an incorrect test prediction, retained below.
A writer probe also accepted a duplicate append with the old-only EF missing
before historical component validation was added. These are real evidence gaps.

Fixes are deliberately limited:

- Independent `packed_history.py` traverses the full externally pinned CURRENT
  parent chain using independent schema/lineage validation. It restores and hashes
  **every distinct historical source version**, not just query/selected roots.
  Decoder, native materialization, structural index checks and full source SHA
  remain mandatory. A metadata size alone never earns budget.
- Packed writer validates all distinct historical physical objects before
  publication and counts all retained files, including unknown/unreachable state.
  Only its own live build staging is temporarily excluded. Projected CURRENT,
  snapshots, decoder and required sidecars count; duplicates/rename earn no bytes.
- Existing Vault hooks default to no-op for V0/4K/stream profiles. Packed storage
  records transaction-owned additions before the existing crash boundaries.
  Under the existing LOCK, restart removes only journal-owned *unpublished*
  additions and that transaction's staging. Published roots/objects are retained;
  protected ancestor objects/roots cannot be deleted. Unknown/legacy staging is
  never garbage-collected and remains charged. No general GC or migration. Publication target is recorded separately from
  snapshot deletion ownership, so reuse of an unowned valid orphan cannot turn
  a committed publication into a cleanup exception. Earlier owned journals remain readable.
- Clean replay accepts absent empty directories (ZIP transport), but refuses persistent staging and unexpected representations in its
  packed-only capsule, so restored source caches cannot be transferred silently.
  Codec decoded-size admission also rejects bool/nonpositive/oversized lengths.

Snapshot/receipt formats, explicit revision increments on rename, limits and
`independent_index_source_semantics=false` do not change. Full-source restore/scan
and structural index checks are not independent reconstruction of every rank/EF
source-semantic relation. Existing raw/deflate/bzip2/xz routing is reused; no real
Precomp, VIKA, browser bridge or new codec is introduced.

## Fixture and actual assertions

Seed20261109. Text524288B, binary262144B (NUL/FF and unique end marker), empty0B;
revision prepends14B. Five snapshots: initial / add duplicate logical doc / rename
stable doc0 (revision1) / revise content (revision2) / revert to old content
(revision3). Four physical content objects including empty, **1310734 historical
unique bytes**, versus **1310720 current logical bytes**. Duplicate/rename/revert
reuse the exact old object files and manifest hashes; old snapshots/file lengths/
SHA256 and15 original query receipts are compared after subsequent imports.

17 query cases cover each snapshot's absent/new revision marker, binary end marker,
high-count pattern, plus document-boundary trap and zero result cap. Counts and
complete coordinates are independently compared to original synthetic byte arrays.
High counts keep OMITTED_HIGH_COUNT/UNTRUSTED, never fabricated enumeration.
Five selected versions restore against SHA256 computed from original fixture
bytes, transported independently of the Vault. Original inputs stay outside the
capsule; rejected tiny/random inputs are retained.

Producer asserts16 rejection cases, two concurrent writers (exact exits0/4,
loser reason WRITER_PARENT_CONFLICT), and all five real SIGKILL boundaries:
object-ready / snapshot-ready / before-current / after-current / published.
Each runs in a separate directory, restarts in a fresh process, verifies old/new
publication state and full history, then forces budget rejection and retries.
A valid preexisting orphan snapshot is separately reused and kept byte-identical.
Old files remain unchanged. Pre-recovery staging/leftovers are inventoried honestly;
post-recovery owned staging is empty. Unknown staging is separately proved retained.

Independent runner checks17 claims, full5-root chain, all4 distinct versions and
five selected external SHA restores. **25 named negative cases** assert specific
exception types/reasons: legacy count/root/missing/corrupt; missing ancestor and
old-only object/component; decoder; wrong external CURRENT; coverage/false empty/
stronger proof/fabricated coordinates; duplicate path/ID/revision/lineage; inflated
ancestor metadata without source; codec/size/bool/decompression cap; required
sidecars and unreachable objects exhausting complete retained budget. Any other
incidental exception fails the test. No required case is optional/skipped/mocked.

## Reproduction and cost evidence

Use a clean disposable checkout, frozen reference at
`c33cb493bc75f13ee4a904fabb190a29a196c94b`, existing environment and fresh outputs:

```bash
source /workspace/glyph-env/activate.sh
cmake -S "$REFERENCE" -B "$REFERENCE/build"
cmake --build "$REFERENCE/build" --target build_sa_binary_v1 build_bwt_binary_v1 -j2
g++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread \
  experiments/loc_text_ef_v0/native_run_stream_inverse_rlb2_ef.cpp -o "$SCRATCH/decoder"
python3 -I experiments/core_l11_integrated_vault_v0/ingest/packed_gate.py \
  --reference "$REFERENCE" --decoder "$SCRATCH/decoder" --output "$SCRATCH/baseline"
python3 -I experiments/core_l11_integrated_vault_v0/ingest/replay_packed.py \
  --capsule "$SCRATCH/baseline/capsule" --pins-json "$BASELINE_PINS" --output "$SCRATCH/replay-baseline"
python3 -I experiments/core_l11_integrated_vault_v0/ingest/packed_lifecycle.py \
  --reference "$REFERENCE" --decoder "$SCRATCH/decoder" --output "$SCRATCH/lifecycle"
python3 -I experiments/core_l11_integrated_vault_v0/ingest/replay_packed.py \
  --capsule "$SCRATCH/lifecycle/capsule" --pins-json "$EXTERNALLY_EXPECTED_PINS" --output "$SCRATCH/independent"
```

Pins are read separately from producer pins.json (not exported in capsule).
The workflow keeps its existing seven jobs and prior gates. Its existing ingest
job exports only lifecycle packed Vault +17 claims; the existing independent
job receives the expected head/query/claim/selected-source hashes separately.
No source arrays, importer oracle or producer success flag establishes acceptance.
Independent process checks it imported no writer/query-producer modules.

Reports emit exact checkout HEAD/tree, every retained file SHA/length/allocation,
category totals, source current/history denominator, actual object counts, parent
pins, preserved receipts, rejection reasons, crashes and commands encoded in the
workflow/CLI. Each import/history/query phase and whole gate record wall/self CPU,
reaped-child CPU, cumulative self peak and max reaped-child RSS separately.
**Max single-child RSS is not process-tree peak**, which is unmeasured. Expanded
view/source file lengths and scratch sums are separate from retained lengths;
they are not sampled peak transient allocated blocks. Native build/source trees,
compiler/Python/stdlib/Git installations are external costs; no installed ratio,
RAM bound, SLA, general compression or power-loss proof is claimed.

Local completed lifecycle (88040B decoder):116386B retained regular file lengths,
188416B allocated regular-file blocks; directory allocation reported separately.
Four object metadata2115B, packed components21032B, five snapshots5134B, CURRENT65B,
decoder88040B. CI must measure its own build; do not require cross-build byte
identity or inherit local PASS. All seven CI jobs and logs at exact final HEAD
are required before ACCEPTED_RESEARCH. Evidence retention30days, capsules14days.

## Failure/fix ledger (local preserved logs)

All logs remain in `/tmp/glyph-packed-lifecycle-z8Kyuo/logs/` (not immutable release
artifacts); hashes below identify these particular observations, not a portable
expected stdout hash:

| Log | Outcome | SHA256 |
|---|---|---|
| baseline-producer.log | original PASS, exit0 | def991c4ecbe901551b5532a99ca44c559d96a2a29165d8ecdcd61345741488c |
| baseline-independent.log | original PASS, exit0; dirty staging later identified | cc097b9a696b5bb26c30675760fa7774600972eb26aad6a670824490b6c70b0e |
| history-falsifier.log | test prediction failed; old budget rejected, exit1 | cc2c4a1e3915fb6d2b6807165fceaf67938f68da8a731e7fb3851bec99c6374b |
| history-falsifier-v2.log | ancestor missing, old replay falsely PASS | b55e17ae75ef1687155508600fe0976c2a0d3ba4e097d4cf4c6b3d6557235605 |
| writer-history-falsifier.log | missing historical EF accepted before fix | a15708bf2625eb3ca7542b21ce781063daa612f48fabdb80e157dcba678f3363 |
| lifecycle-attempt1.log | fixture repeated CRASH_ONLY, expected1 was wrong; exit1 | ad7d72f6e1a49a2c42f8644597b0fa6f2b1b413f1e94de5e5dcc50221f5e5ed9 |
| preexisting-snapshot-falsifier.log | first candidate57b2 published then raised on unowned snapshot reuse; fixed target/ownership separation | a02329f6b40daa640d72b0ec4acc09bbd16370df146b45048e9655e17e577e57 |

First candidate57b2 research run42613999237 has six successful jobs and a
bounded-file-independent FAIL: ZIP transport omits empty vault/staging and the
initial checker incorrectly required its directory entry. Required data was not
missing. This failure is retained at
https://github.com/yasha1971-coder/glyph-engine/actions/runs/42613999237/job/129151990340.
The next candidate reruns all seven jobs, not a relabelled old green.
Artifact transport drops empty staging directories; replay treats absent staging
as empty while still rejecting every persistent staging entry.
The CRASH_ONLY marker was made unique; no count algorithm was weakened. Later complete
producer and independent runs pass. Historical baseline source caches remain
preserved; new copies remove only their newly journal-owned transaction artifacts.
CI results/final commits and unresolved limits are delivered in PR29/response;
coordinator owns ANCHOR/CURRENT_TASK/append-only CONTEXT updates. No memory update
is fabricated. Stop after delivery; no merge or follow-on engineering is authorized.
