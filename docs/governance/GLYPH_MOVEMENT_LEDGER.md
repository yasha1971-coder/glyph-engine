# GLYPH Movement Ledger — additive branch record

This path was absent from the `glyph-v2-1tb-truth-gate` base. This additive entry
does not replace the historical Centenary branch ledger or claim its merge.

## 2026-09-08 — chunk state single-writer gate

Base: `3b03c3805133005bf0c0e0054a4d8368e61719b0`.

The existing valid-hex checkpoint corruption fix was recovered and verified
before development. Baseline: 14 local tests passed, including 10 chunk tests;
published CI run `34273911487` succeeded.

The next bounded change enforces the documented single-writer precondition for
cooperating Linux/WSL chunk CLI processes. It adds an advisory process lock held
through database close and receipt publication. Busy callers return 75 before
database access. Subprocess tests cover a held lock, a paused actual CLI, normal
exit, and SIGKILL release. The new tests fail against the base in three cases;
the corrected component suite passes 16 tests (12 chunk, 4 metadata).

No source inventory was rerun on user data. No frozen V1 artifacts, immutable
tags, canonical Vault, metadata scanner, payload format or compression algorithm
were changed. TB readiness, snapshot atomicity and hostile-writer exclusion are
not claimed. See `experiments/personal_memory_1tb_v2/CHECKPOINT_VERIFICATION.md`.

## 2026-09-09 — V2 convergence boundary and Composition Foundations

Base: `17b59df03bda8c3a9eb9ac027a59dd5f453824d6`.

The two unique commits from `composition-foundations-v1` were integrated as the
accepted D-007 precondition specification. This does not enable Index Forest or
change the Composition V1 wire contract.

`docs/architecture/GLYPH_V2_CONVERGENCE_MAP_V1.md` records the 18 remote branch
heads, their Git reachability from the V2 base, the 13 immutable tag names, the
public claim boundaries and the intended status of each target capability.

The public V1 claim remains verifiable exact-byte evidence over committed
corpora. Append-only Personal Memory is expressed as immutable generations over
that kernel. Hybrid routing is a physical optimization and cannot alter corpus
identity, document order, canonical coordinates or restore bytes.

The prior Personal Vault ratio `0.335` is explicitly `UNVERIFIED` until a
repeatable receipt with corpus identity, complete byte accounting, parameters,
artifact hashes and independent restore is located. No Disk D scan, compression
run, payload implementation, branch deletion or public-site change occurred in
this movement.
