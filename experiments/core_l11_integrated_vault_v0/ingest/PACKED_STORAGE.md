# Packed storage acceptance / research

VaultPacked reuses VaultScale logical STREAM_V2 snapshots and exact query contract.
Physical storage is explicitly opt-in: root.json plus storage.json and three
*.packed components. Ordinary readers fail closed on missing unpacked components.
No migration or replacement of V0/4K/frozen/deployed Vaults. Select a new path.

Recovered encode_best chooses actual raw/deflate/bzip2/xz size for each physical
component. The original physical root pins decoded lengths and SHA256. Packed
codecs are not trusted flags: decoding must reproduce those exact pinned bytes.
No additional source archive is retained. Precomp isn't used on native indexes.
Decode has output caps, component128MiB and XZ memory256MiB caps. This is not
hostile-process isolation or an adversarial filesystem threat-model closure.

Before replacing CURRENT, the writer checks ALL retained regular file lengths
including decoder, packed indexes, metadata, all snapshots and projected CURRENT.
Budget counts distinct authenticated source-content versions across retained
history once, so duplicate logical slots cannot inflate the budget. Build staging
is temporary and excluded; installed code, allocation and peak scratch costs
remain separate. Compact product acceptance does not promise total installed cost.

On AcceptanceRejected, the writer removes only additions from this transaction,
under the existing lock, fsyncs affected directories and preserves old CURRENT.
Generic crashes retain the preexisting research retry/orphan semantics; arbitrary
power loss/torn writes are not proven. Old profiles' default acceptance hook is
no-op. No claim that their earlier expanded results meet the compact requirement.

Queries temporarily expand indexes and use the unchanged streaming indexed query.
Independent proof expands the same bounded physical representation then executes
independent_scale full byte restoration/hash/scan. independent_index_source_semantics
remains false. This does NOT claim independent reconstruction of every rank/EF
relation, compact proof or instant startup. Whole query view currently expands
all retained objects, including history: transient time/RAM/disk cost is open.
No production replacement for the existing TRUST2 touched-block interactive path.

packed_gate.py exercises two snapshots, old-byte preservation, exact indexed
query and independent full replay, actual SIGKILL before publication and retry,
corrupt packed component rejection, first import over budget and oversized append
rejection without changing the old retained tree. Synthetic data only. Full package
measured at two snapshots; no original source file retained in accepted Vault.
Reproduce with --reference /ABS/REF --decoder /ABS/DECODER --output /ABS/NEW.

Next: separate-runner packed-only capsule replay with transported root pin;
broader append/revision/dedup/crash negatives and per-stage RAM/CPU/temp costs;
then bounded heterogeneous source classes, avoiding unconditional full opaque
media indexes. Current packing is a compatible bounded research step, not final
heterogeneous preservation routing or a real-VIKA compactness claim.

## Separate-runner replay continuation

packed_gate.py exports only capsule/vault and three claim JSONs; pins.json stays
outside the capsule and is transported through CI job outputs. Independent job
imports no Vault or producer. It authenticates claim pins and snapshot roots,
expands checked components, independently restores/scans all bytes for old absence,
new hit and high-count cases. Complete retained budget is rechecked. Wrong count,
wrong external pin, missing and corrupt component reject. Original source paths
are not transferred. Test-copy mutations are restored before completion.
This covers this two-snapshot synthetic fixture, not arbitrary history, real VIKA
or lower temporary expansion costs. Local passes do not establish cross-runner
acceptance until the separate CI job at the exact new HEAD passes.

## Lifecycle reconciliation — 2026-11-09

The preceding separate-runner paragraph describes the earlier claim. A baseline
reproduction found retained restored-source staging after its SIGKILL, and a
missing ancestor-only object escaped old query-selected budget verification.
Do not repeat the former no-source-transfer claim for that historical capsule.
It remains preserved at its checkpoint. New packed transaction-owned restart
cleanup and complete-history restore/hash verification address those findings;
clean capsules reject staging and unknown representations. See
[PACKED_LIFECYCLE.md](PACKED_LIFECYCLE.md) for failures, five snapshots/revisions,
five process-crash/restart/budget-retry cases, exact commands and cost boundaries.
No historical artifact or format was overwritten; unknown staging is never GC'd.
