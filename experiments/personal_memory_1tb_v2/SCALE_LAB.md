# Scale lab V1: streaming storage endurance

This is a separate experimental sharded object store, NOT the Personal Memory
application. It does not remove that application's 8MiB/10,000 paths/1,000 history
limits and does not demonstrate a unified 1TB catalogue, version history or GUI.
The reused GLYPH component is verified_hybrid_archive.decode (raw/deflate9).
SQLite catalog, deterministic workload generator and resume driver are lab code.

Default target is exactly 1,000,000,000,000 logical bytes, maximum request 2TB.
Blocks are 16MiB with a short final block. Half of input is unique SHAKE256 output
stored raw. One quarter is repeated local samples, one quarter replaces 4KiB per
64KiB with deterministic artificial content; these use deflate9 or raw fallback.
Identical complete blocks share an object. Ratios MUST NOT be advertised as
compression of 1TB of real personal files. No bzip2/xz/Precomp comparison here.

Only up to 64 distinct 1MiB prefixes from permitted file types are copied from
explicit source folder. Discovery bounded at depth 6 / 10,000 entries. Password
file exact names `тут все.txt` and trailing-backslash variant excluded. Other seed
content remains private: NEVER upload seed files, objects or the source folder.
No automatic uploads. Source files are never deleted or edited. Archives outside
the source folder and old VIKA manifests are not used.

New output only; fixed 300GiB free-space reserve, 1GiB process address space limit.
An external process can consume disk after the reserve check: actual ENOSPC fails
the run. Each object is fsynced and read back before the SQLite progress commit.
A final full pass reconstructs deterministic expected input, verifies index and
exact output bytes. fsync under WSL/DrvFS is not proof of physical power-loss
persistence. Reads may be OS-cached. Index is local trusted state, not signed.

Resume with same arguments plus --resume. It validates ALL completed blocks
before continuing. Objects published before an interrupted index commit are
verified/reused; .partial files are not counted. Resume after seed sampling failed
before config creation is not supported. Do not delete partial output blindly.

RESULT.json reports logical workload, unique payload and filesystem allocation,
per-kind totals, summed encode time, summed generation+encoding+fsync+readback time
(excluding SQLite commit), final verification time and peak RSS. These timings do
not constitute full end-to-end wall duration across interrupted sessions.

Typical command (stop previous test before resuming; one writer enforced):

    python3 experiments/personal_memory_1tb_v2/scale_lab.py --source /mnt/d/VIKA_proekt --output /mnt/d/GLYPH-TESTLAB-1TB-V1 --target-tb 1

Keep laptop plugged in and awake. Completion may take many hours: use progress
measurements for estimates rather than assuming earlier Silesia throughput.
Tests exercise small real files, final tail, restart, orphan object, corruption,
reserve rejection and excluded seed. They do not prove 1TB completion or a real
hardware crash. RESULT is only emitted after the actual full-target verification.
