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

## 2026-09-10 — reversible precompression truth probe

Base commit: `0ca6cf86695d8bc942081642cc081be54d8df7b5` on the unmerged
`codex/glyph-v2-hybrid-compression-v1` line.

The prior whole-file pilot shows that work on already-compressible text/code
classes alone cannot establish the 30% product target. The next bounded question
is whether already-compressed media and container streams admit reversible
normalization before the ordinary lossless router. No private-corpus class
distribution or measurement is recorded by this movement.

`GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1` adds an explicitly
experimental route for JPEG, PNG, PDF and ZIP. It pins the external executable
by SHA-256, gives it only temporary copies, isolates each invocation in a fresh
working directory and selects its output only after immediate exact-byte
reconstruction. Unsupported, crashed or mismatching attempts fall back to the
ordinary verified router and remain counted in the receipt.

The reference dependency is upstream Precomp v0.4.7 source commit
`31b693d843e378e7d30190736f95c095769868b0`; no third-party executable or source
is vendored. The observed upstream Linux x86-64 release binary SHA-256 is
`6a5289ba81ea658e8e7984bf32791635f53b8e7cee4aafc26b853c7cd5b018f8`.
The observed release ZIP SHA-256 is
`fcd308310135cdc1ae25b45288bcc5201f9cabf5a9e707cfb33b0f99c548de0b`.
This identifies tested bytes but is not a signature. The upstream binary labels
itself development/test-only, so this gate may establish compression potential
but cannot establish the final Personal Memory trust boundary.

No VIKA result, 30% claim, Disk D scan, public-site change, or ACEAPEX change is
made by this movement. Production use remains blocked on reproducible source
build, hostile corpus testing, dependency/license review and an independent
full-corpus restore receipt.

## 2026-09-12 — Recover existing Laptop V1 before V2 integration

Status: source review and continuity correction, not new product acceptance.
Local experimental adapter checkpoint: 109f52411c2e3b216df3609e9515567a697fc098;
44 synthetic tests, not integration of the existing V1 trusted reader.

Read the existing portable V1 README/profile and laptop/OVH evidence. Executed
`python3 deploy/glyph-v1/glyph.py verify-profile`: GREEN, 5 frozen files and
6 evidence files verified. No live host replay or model inference performed.

The old V0.1 launcher is not the complete Laptop V1 implementation. Preserve
frozen discovery, explicit human selection, authenticated materialization,
canonical meaning before model prose, and the prior Local Intelligence work.
The four previously compared wrapper/CLI/restore files did not cover the
bounded-memory builder fix; current experimental builder still has global
run accumulation. Recover that exact implementation before legacy real ingest.

Private continuity records were updated additively in GLYPH/CONTEXT.md and
GLYPH/ANCHOR.md at yasha-context commit
4be5595dbde502387961e3ff8208ac5929ac39f7, with prior facts and access limits.
BREAKTHROUGH_LAW.md remains unchanged. No unrelated project files changed.

Next implementation target is isolated compressed preservation integration with
the existing trusted read/selection contract, complete size accounting, version
and crash/corruption gates. No installed release, frozen evidence, canonical
Vault or model bytes changed. No iPhone, production or full-vault ratio claim.

## 2026-09-12 — Experimental compressed backend connected to existing Phase B

Added an optional preservation backend to the portable host, leaving frozen files
and default V1 cache behavior unchanged. The separate experimental V2 entry point
uses existing human-selection and authenticated-object-map checks, requires exact
path/size/digest equivalence, and host-rechecks decoded output before publication.
V2 receipt identity is explicit. Six new synthetic tests and six existing portable
tests passed; full V2 suite now has 50 tests. Host authentication is stubbed in
three new sequencing tests; Precomp fixture is fake. No live corpus/host proof.
Details, commands, failure history and limitations: COMPRESSED_PRESERVATION_GATE.md
in experiments/personal_memory_1tb_v2. Full search-index retention still means no
total-space reduction can be claimed. No installation, private-data publication,
source deletion, frozen tag movement, model invocation or ACEAPEX change.

## 2026-09-12 — Non-mocked authenticated preservation fixture

Added valid tiny on-disk RLB3X/LOC2/rank/AUTHLOC fixtures and seven tests using
the frozen real reader and real Phase B host. No authentication function mocks.
Exact search/absence and index, map, root, external pin and compressed payload
corruption are checked. Synthetic Phase A is not natural-language acceptance.
One 32-byte synthetic corpus is not evidence of VIKA compactness, scaling or
complete V2 integration. The old full search representation remains required.
No installed C/D copies, frozen sources, external services or other projects changed.

## 2026-09-12 — Local companion browser pilot for existing compressed archive

Added loopback-only name/path browser with explicit per-file download and hash
verification. No re-ingest, no source rewrite, no automatic selection or LLM.
Three HTTP tests and an actual pinned Precomp synthetic JPEG worker roundtrip
passed. Worker has resource limits, not a hostile-code sandbox. This is an early
user-facing companion, not full Vault integration or preservation/search dedup.
Details: experiments/personal_memory_1tb_v2/LOCAL_BROWSER_PILOT.md.
