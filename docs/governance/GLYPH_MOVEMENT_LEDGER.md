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

## 2026-09-09 — fixed-chunk dedup baseline gate

Base: `256f4436cf709bfc129b5c02fb411f518bcfcbc3`.

STATUS: code-proven on bounded fixtures; real-corpus measurement remains open.

`GLYPH_DEDUP_BASELINE_V1` measures only exact reuse already established by a
completed fixed, file-aligned chunk-truth state. It validates the canonical
receipt and checksum plus all SQLite chains, counters, content roots and the
manifest root before reporting unique bytes, reused references, exact duplicate
files, storage ratio and saving fraction. The output is deterministic for an
unchanged state and carries explicit non-claims for compression, shifted-content
deduplication and independent restore.

The test matrix includes exact duplicate files, deterministic receipt replay, a
shifted-content negative control, incomplete state, receipt corruption, valid-hex
database corruption and a busy writer. No user corpus, Disk D inventory,
ACEAPEX path, payload format or public surface is changed by this gate.
This implementation alone does not close ordered convergence gate 3; that
requires a receipt from a named real corpus, starting with the VIKA pilot.

## 2026-09-09 — verified hybrid compression truth pilot

Base tree: `db0a004c92ec701bd5bf184f0e46ebd493e4c084` from remote commit
`8856a2234ccb37ac93ee5a4b79f432d59e2475fc`.

The fixed-chunk VIKA matrix is accepted only as user-executed transcript
evidence: 415 files, 249,967,975 logical bytes, all six chunk states complete
with zero errors. Gross fixed-chunk saving ranged from 2.556308663% at 4 KiB to
0.083646315% at 1 MiB and 8 MiB. It is not compression evidence and does not
meet the product target.

`GLYPH_VERIFIED_HYBRID_ARCHIVE_V1` creates a real content-addressed payload
archive. Complete-file SHA-256 identity supplies exact-file deduplication; each
unique object is trial-compressed with raw, DEFLATE-9, bzip2-9 and XZ-9 and the
smallest byte representation is stored. The numerator includes object payloads,
the canonical manifest and checksum sidecar. The separate restore operation
rejects corrupted objects and verifies restored bytes and the complete manifest
root before reporting success.

The VIKA acceptance target is at least 30% logical-byte reduction with
independent byte-perfect restore. No result is claimed until a named receipt and
restore evidence are produced. This pilot does not implement CDC, delta,
cross-file dictionaries, RLBWT queryability, permissions/timestamp preservation,
or the final append-only generation protocol. ACEAPEX is outside this movement.
