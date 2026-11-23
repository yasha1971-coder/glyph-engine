# Packed search cost — bounded research gate V0

Continuation of existing Issue28 / Draft PR29 from
`df740481378511d5c6f88865d4203a3722854c40`, tree
`5c86af3e12859a65d7869252d9b292cede23005b`. Existing default query, storage,
receipts, source/history budget, native RLB2/RLR2/EF implementation and independent
verifiers are unchanged. Additive research adapter + measurements only. No merge,
new Issue/PR, main/frozen/CONTROL/OVH/private data, server/GUI/model or format migration.

Canonical memory still calls the completed lifecycle acceptance pending and retains
older stronger no-source assertions. Actual baseline CI42615078709 is7/7 SUCCESS;
PACKED_LIFECYCLE.md preserves the corrected source-cache/history findings. These
are reported discrepancies, not permission to rewrite memory/history. Coordinator
owns canonical review. Original product freeze criterion remains unresolved;
this experiment does not invent a product SLA or declare CORE frozen.

## Lock and hypothesis before measurement

LOCK.json was written before profiling or candidate measurements. SHA256:
`0c03163856e8f66d297f000c0238098979a58f9b785eb9677471bb1ecbc1d2c8`.
The exact input implementation is pinned by the baseline commit; source audit
must show every preexisting file except this workflow unchanged.

Separate preliminary original-query cProfile:0.486546s for binary end marker,
three QueryState initializations0.383s (~79%), including native materialization
0.197s. These cumulative intervals **overlap**, not additive charges. It justified
an opt-in session experiment, not a new backend. Attribution is profiled separately
again; no profiler runs in latency comparisons.

Fixed hypotheses: QueryState startup >=50% of original query profile; median
session-batch wall / original same-process baseline-batch wall <=0.8. A miss must
remain a negative measured result, not trigger threshold/corpus tuning. Correctness,
coverage, resource checks and independent replay are mandatory regardless of speed.
No requirement to beat raw grep or any competing engine was invented.

## What is measured

Existing packed_lifecycle fixture: initial text/binary/empty, duplicate, rename,
revision, revert; five snapshots, four content versions;1310734 distinct historical
source bytes. Main workload is all five separately pinned CURRENT cases: absent
revision marker, one binary end hit, high-count GLYPH, document-boundary trap, cap0.
Four fixed cycles =20 requests per batch. Three trials rotate a declared order:

1. fresh_process / baseline_batch / session_batch;
2. session_batch / fresh_process / baseline_batch;
3. baseline_batch / session_batch / fresh_process.

fresh_process starts an entirely new Python worker for **each** request. Cold
response includes process/import/startup/query/serialization; whole process exit
is additionally charged. This is **not disk-cold**: OS page cache is uncontrolled,
not flushed. baseline_batch repeats the unchanged VaultPacked.query in one process;
it still expands and creates/authenticates new leaf states every call. session_batch
pays process/import/session startup once plus all20 requests, guards, serialization
and close. Report first answer = readiness + first request, per-query round trip,
whole batch, CPU and memory separately. All trials/results are retained; no fastest
run selection. Common whole-history authentication/admission and setup run outside
all three search timings. Independent verifier costs are separate: producer latency
is not end-to-end independently verified application latency.

## Session boundary, not a silent cache of success flags

One explicit root, <=4 distinct active nonempty objects, <=8MiB active unique source,
<=32MiB conservative admitted temporary file lengths. Existing root/empty/decoder/
component checks and unchanged QueryState authenticate exact physical source at
open. Duplicate logical docs share physical QueryState but remain distinct IDs,
revisions/count/coordinates; no logical deduplication. Full-history physical view
is expanded once, including old/unreachable objects, and paid as temporary data.
No new retained index/sidecar or raw-source cache is written into Vault.

Every request hashes **all** live Vault files, private expanded inputs/state copies,
and declared source-code dependencies before and after query, and rechecks the
frozen reference HEAD on both guards. This still costs
O(retained bytes + files); it is not a succinct or sublinear proof. Any missing,
changed, added, private-corrupted or code-changed input fails closed. The underlying
source-auth result is reusable only while these immutable bytes match, with the
same unsigned root trust assumptions. No source scan is silently reclassified as
full independent index semantics; receipt remains STREAM_EXACT_RECEIPT_V2 with
independent_index_source_semantics=false and original LIMITED/high-count behavior.

The archive must be quiescent. Any append/CURRENT or file-tree change invalidates
this session, even for an old pin; caller closes/reopens. No implicit retargeting.
This is a **stricter opt-in admission condition**, not a change to the default API.
Real append mutation tests prove old default query and reopened old-snapshot session
still work. Promotion/concurrency policy remains an owner decision. No network
service, root-currentness signature, daemon or production cache is introduced.

## Resource boundary

Each measured worker has its own process group, RLIMIT_AS1GiB **per process**, and
120s timeout. Parent samples group-summed /proc RSS and temporary file lengths every
10ms. Observed RSS >256MiB kills **only that owned group** and fails the run.
A real1-byte budget test verifies the kill/reason, not a mock worker.

Group RSS sum double-counts shared pages, is not PSS/unique memory, and sampling
can miss transient peaks. Thus this is an enforced **sampling guard**, not a
continuous kernel-enforced aggregate physical RAM limit. Per-process virtual limit
is different. Record sampled tree max, OS cumulative self/max-child peaks, self/
reaped-child CPU and monitor costs honestly. Native code/build/installed Python/
stdlib/Git/compiler/source trees remain dependencies; no zero-dependency package,
hardware maximum, process-tree unique peak or fixed SLA claim.

The private state, full expanded view, native nested component snapshots, transient
source authentication and sample map are paid. Constructor's temp admission estimate
is conservative for these input file lengths, not a peak allocated-disk theorem.
Sampler also reports actual observed temporary file-length maximum. Worker close
must leave no temporary files. Allocated Vault files/directories remain separately
inventoried; retained Vault hashes/lengths/allocation must be identical after reads.

## Correctness and hostility

Every answer is canonical-byte compared with old separately pinned claims, not
only a count flag:5 cases x4 cycles x3 trials x3 modes =180 full replies. Default
old roots/limits remain unchanged.17 named cases reject exact expected reasons:
active/history-only/empty/decoder/CURRENT/root missing or changed; unknown addition;
private EF; wrong expected pin; closed session; temp/object/source admission;
fully re-pinned foreign RLB3X magic; actual append invalidation / no retarget;
real RSS-budget group kill. Historical revision replay and old API/reopen positives
are also executed. Wrong magic is a negative input, not an actual RLB3X reader or
cross-format compatibility result. Never plug an RLB2 decoder into RLB3X/LOC2.

Gate exports only copied packed Vault +17 existing claims; the five current claims
are replaced with **actual session-produced bytes** whose SHA equals old expected
pins. No producer metrics/success flag/source oracle is in this capsule. Expected
head/query/claim/selected-source pins go separately via job output. Existing clean
independent job uses unchanged replay_packed.py: all5 parents,4 content versions,
17 queries, five external-SHA restores and25 source/history/coverage/codec negatives.
All previous seven jobs remain required. No claim that VIKA113–124s observations
apply here; that benchmark used a separate RLB3X/LOC2 runtime and different corpus.

## Reproduce

Use existing environment, frozen reference C33 and an existing fresh lifecycle
capsule/pins from packed_lifecycle.py; all outputs in disposable workspace:

```bash
source /workspace/glyph-env/activate.sh
python3 -I experiments/core_l11_integrated_vault_v0/ingest/search_cost/gate.py \
  --capsule "$LIFECYCLE/capsule" --pins "$LIFECYCLE/pins.json" \
  --reference "$REFERENCE" --output "$SCRATCH/search"
python3 -I experiments/core_l11_integrated_vault_v0/ingest/replay_packed.py \
  --capsule "$SCRATCH/search/capsule" --pins-json "$EXTERNALLY_EXPECTED_PINS" \
  --output "$SCRATCH/search-independent"
```

Original helper imports mutate sys.path. Inspection found `from hostiles import run`
would resolve the old composition hostiles module; the new helper was renamed
cost_hostiles before the combined gate, without changing a checker or threshold.
Local initial timing-only run and final combined run, attribution profiles and
independent replay are preserved under /tmp/glyph-packed-search-2i8Zdy/logs and work.
No semantically wrong answer or waived hostile failure was accepted. Early combined
ratios0.13135/0.12199/0.13730 are retained. The publication run also pays frozen-HEAD
rechecks and client CPU telemetry: ratios0.16420/0.14692/0.18264, startup fraction
0.84421. Both meet the same unchanged prelocked hypotheses on this runner/fixture. CI must measure its own values, not inherit local green.

This is bounded performance research. Unknown product freeze criteria, general
heterogeneous behavior, large histories, full installed/peak transient/unique
process-tree memory, power-loss, unsigned-root freshness and independent full
rank/EF semantics remain open. Old six backend SKIPs / bounded RLB3X builder patch
lineage are not closed. No full top-level VERIFY or production promotion follows.

## Parallel full-SHA scheduling experiment

Owner instruction on 2026-11-23 resumed bounded autonomous engineering. It does
not establish an external customer or alter the proof contract. Existing serial
Session and default VaultPacked query remain byte-identical to their baseline.

`parallel_session.py` is an explicit opt-in subclass. Two read workers call the
unchanged `session.catalog` over every live and private root. Every byte is still
SHA-256 checked before and after each query; frozen HEAD and code checks remain.
No stat cache, cached success flag, skipped history, new retained index, changed
receipt or reduced coverage is introduced. On failure, all reads finish before
private files can be cleaned up. The original cooperative, quiescent-filesystem
scope remains; this does not add concurrent mutation or query/close guarantees.

`PARALLEL_GUARD_LOCK.json` fixes the corpus, three rotated orders, resource
arguments, correctness conditions and public guard-time ratio hypothesis <=0.8
before measurement. Three current public receipts repeated three times per
batch, three batches and two modes give 54 pinned replies. The earlier lifecycle
fixture gives 120 pinned replies across the same three orders. A missed speed
hypothesis remains a negative result, separately from integrity acceptance.

Local measured code checkpoint: `1727b245da1721b8073bd1117f9e39906e0ceb1f`.
Public paired guard ratios: 0.597585 / 0.587132 / 0.592668; median 0.592668.
Warm serial / parallel medians in milliseconds: hit 686/558, absence 343/214,
high count 323/198. Open once per arm: 33.480/34.679 seconds. Every request
hashes 466,016,288 live/private file bytes, plus declared code. Thus the number
of checks is fixed, while their byte cost scales with state. This is a CPU
scheduling benefit, not less verification work or a new search algorithm.

The small fixture's median guard ratio was 1.303417: guard work is slower there.
It does not justify an automatic routing threshold or default promotion. These
measurements use ordinary OS caches and in-process calls; they are not directly
the old stdio round-trip timings. Two open private states share the measurement
process; cumulative process peaks are not isolated per-arm RAM measurements.

The probe reuses the original hostile assertions with the real candidate class
injected: 16 constructor/session cases use the candidate. The original RSS
assertion still launches its original worker and is reported as such; an
additional candidate-owned worker is actually killed at a one-byte RSS budget.
Seven additional file-integrity/failure-cleanup tests use real reads, including
same-size content mutation with restored mtime and a pending read after another
catalog fails. Scheduling instrumentation does not substitute hash results.

```bash
python3 -m unittest discover -s experiments/core_l11_integrated_vault_v0/ingest/search_cost/tests -v
python3 -I experiments/core_l11_integrated_vault_v0/ingest/search_cost/parallel_guard_gate.py \
  --capsule "$LIFECYCLE/capsule" --pins "$LIFECYCLE/pins.json" \
  --reference "$REFERENCE" --output "$SCRATCH/parallel-guards" --profile small --hostiles
python3 -I experiments/core_l11_integrated_vault_v0/ingest/replay_packed.py \
  --capsule "$SCRATCH/parallel-guards/capsule" --pins-json "$EXTERNAL_PINS" \
  --output "$SCRATCH/parallel-independent"
```

Public `--profile public` accepts only the previously pinned three-case, whole
100MB enwik8 capsule. It explicitly requests 100,000,000 source bytes and
512MiB temporary admission through the existing Session API. The accepted old
8MiB/32MiB defaults and lock are unchanged. It is dev evidence, not a fresh
holdout, enwik9 admission, a 200ms SLA or independent cryptographic completeness.
Independent replay still restores and scans source. Old six backend SKIPs and
bounded RLB3X builder lineage remain open.

The source-count I0 reconciliation is a separate preserved movement in
`docs/governance/GLYPH_MOVEMENT_LEDGER.md`. Historical paragraphs above describe
their own earlier checkpoints, including their former top-level VERIFY scope.
