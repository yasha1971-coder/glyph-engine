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
