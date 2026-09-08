# Checkpoint verification correction

Base: 57da5c72334e42e049eb3d820918a9889595a0ff.

CLAIM: Saved chunk hashes are checked against source bytes before continuation, including completed files. Coverage, counters and completed content roots are checked. Files rejected by verification are excluded from duplicate statistics.
STATUS: locally tested; not a full GLYPH release or large-corpus validation.
EVIDENCE: python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v
Result: 10 tests passed. With the new tests against the base implementation, seven subcase failures reproduced false success: partial/completed valid-hex digest corruption, deleted final chunk, wrong root, wrong offset/count, and changed prefix bytes with restored mtime.
FALSIFIER: Any of those mutations returns exit 0 / complete=true.
SUPERSEDES: Format-only SHA validation and unconditional reuse of done files.

## Scope and cost

Each invocation re-reads all saved checkpoint bytes. --max-chunks bounds NEW chunks only; verification is additional I/O. The stdout field checkpoint_bytes_verified_this_run reports successfully compared bytes. This is a correctness baseline, not efficient TB-scale resume. Memory remains chunk bounded.

Requires stable source files, a trusted metadata inventory and a single writer owning the state directory. This does not provide an atomic source snapshot, a lock against competing writers, protection against coordinated source/state tampering, rollback protection or an externally authenticated root. Receipt SHA files are checksums, not digital signatures. File content roots are chunk-list commitments, not ordinary whole-file SHA256.

No compression, payload storage, CDC, restore or incremental snapshot implementation is added. No real user corpus was read in these tests. Do not infer full-product readiness from this regression result.

## 2026-09-08: enforced cooperating single-writer boundary

Base: 3b03c3805133005bf0c0e0054a4d8368e61719b0.

CLAIM: Cooperating Linux/WSL CLI processes serialize access to the chunk state.
The lock covers database initialization, import, verification, hashing and receipt
publication, and is released after database close or process death.
STATUS: locally measured on Linux; no TB-corpus run.
EVIDENCE: `python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v`
returned 16 tests OK (12 chunk tests and 4 metadata tests). Before the change,
the two new test methods failed in three cases: pre-held lock, concurrent CLI
with normal-exit scenario, and concurrent CLI with kill scenario. The corrected
tests verify busy rejection before database creation, rejection while an actual
CLI is paused, and successful continuation after normal exit and SIGKILL.
The process pause occurs before file hashing; this test does not claim arbitrary
mid-transaction power-loss durability. Existing chunk-checkpoint tests still run.
FALSIFIER: A second cooperating invocation enters the database while the first
holds the lock, or the state remains locked after the owner process dies.
SUPERSEDES: The unenforced one-writer precondition for cooperating CLI processes.

The earlier checkpoint re-read cost and trusted-inventory/stable-source limits
remain. The lock is advisory, not protection against hostile or older writers.
