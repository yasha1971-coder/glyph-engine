# GLYPH_EMBEDDED_I0_PREFREEZE_REVIEW_DISPOSITION_V1

Status: accepted remediation plan
Date: 2026-07-14

## Reviewed state

Reviewed commit:

    064ba0de88fed74ed28a16c37f4a78028c2d8eec

Review package SHA-256:

    44602521f134d6908a8d14182e1a1ae210a94de6035481bc262dc36a74bf460b

External verdict:

    CHANGES_REQUIRED

Current contract status remains:

    DRAFT_NOT_FROZEN

Runtime implementation is prohibited until a second review returns
FREEZE_READY and the executable I0 freeze gate passes.

## Accepted freeze blockers

The following findings are accepted as requiring resolution before freeze:

- I0-R-001 — mmap validation and served-byte identity;
- I0-R-002 — deterministic doc_id domain and assignment;
- I0-R-003 — immutable query state versus mutable lifecycle state;
- I0-R-004 — complete caller-barrier close semantics;
- I0-R-005 — flag-gated future semantic fields;
- I0-R-006 — index format and runtime profile exposure;
- I0-R-007 — canonical source-path validation;
- I0-R-008 — null-handle and null-directory status mappings;
- I0-C-001 through I0-C-004 — cross-file contradictions;
- I0-G-001 through I0-G-006 — checker coverage gaps.

The following findings are accepted as mandatory pre-freeze clarity fixes:

- I0-R-009 — max_results/capacity violation returns GLYPH_E_ARG;
- I0-R-010 — V1 output flags are zero and future unknown bits are ignored;
- I0-R-011 — doc_offset is a zero-based first-byte offset;
- I0-R-012 — signed-statement encoding freezes before signature implementation;
- I0-R-013 — status codes form one append-only cross-version registry;
- I0-R-014 — V1 open has no deadline.

## Normative design decisions

### Mmap and local filesystem trust

The default V1 profile is:

    GLYPH_TRUSTED_IMMUTABLE_LOCAL_FILESYSTEM_V1

Publication immutability must begin before an index becomes available to open
and must continue for the full lifetime of every handle.

The runtime will:

1. open one root directory descriptor;
2. resolve payloads relative to that descriptor;
3. reject symlink and out-of-root resolution;
4. open and fstat each regular payload;
5. enforce resource limits;
6. create the final read-only mapping;
7. hash and structurally validate through that mapping;
8. re-fstat as a best-effort mutation check;
9. publish the handle only after all checks pass.

Map-then-verify does not provide protection against a writer that is permitted
to mutate the underlying file during handle lifetime.

Hostile-local-writer protection is outside the default V1 profile and requires
a stronger mechanism such as fs-verity, dm-verity, or a future verified-paging
profile.

### Document identity

doc_id is dense:

    0 <= doc_id < document_count

Assignment is exactly the committed canonical manifest order.

Identical corpus identity and manifest identity produce identical doc_id
assignment.

Enumeration is performed by iterating:

    0 .. document_count - 1

An out-of-range doc_id returns:

    GLYPH_E_ARG

doc_offset is the zero-based byte offset of the first matched byte within its
document.

### Lifecycle and close

All query-relevant handle state is immutable after successful open.

The only permitted mutable shared handle state is a data-race-free atomic
active-operation count.

V1 uses a caller-side close barrier and no persistent library closing latch.

The caller must ensure:

- no new operation begins after a close attempt starts;
- exactly one thread calls close for a given handle variable;
- concurrent close calls on the same handle variable are a caller contract
  violation.

When active operations exist:

    glyph_index_close_v1() returns GLYPH_E_BUSY

An E_BUSY close:

- does not alter the handle;
- does not establish a persistent library-side barrier;
- permits the caller to resume operations or retry after draining readers.

### Public metadata

The first eight bytes currently reserved at offset 88 in
glyph_index_info_v1 will become:

    uint32_t index_format_version;
    uint32_t runtime_profile_id;

The total structure size remains 120 bytes.

The values are validated from the opened runtime index and returned to the
caller.

### Source paths

Committed source paths must be:

- non-empty;
- relative;
- free of byte 0x00;
- free of empty, "." and ".." path components;
- free of leading and trailing separators;
- unique as raw byte sequences;
- permitted to contain invalid UTF-8.

A violation returns:

    GLYPH_E_FORMAT

Returned paths remain raw, non-NUL-terminated byte sequences.

### Versioning

A future field that changes safety-relevant or semantic behavior must be
enabled by a new flags bit that an older library rejects as unknown.

Ignored tail fields must be advisory only.

V1 metadata output flags are zero.

Future output flag bits may be added, and callers must ignore bits they do not
recognize.

### Arguments and statuses

A null index_directory returns GLYPH_E_ARG.

A null index handle returns GLYPH_E_ARG for every handle-taking function.

max_results greater than coordinate_capacity returns GLYPH_E_ARG.

Status values form one append-only registry shared by all future ABI versions.
Assigned values are never removed, reused, or renumbered.

### Timing

V1 performs full-payload verification during open.

V1 exposes no open deadline.

Callers requiring bounded open latency must constrain max_mapped_bytes or run
open outside a latency-critical serving thread.

### Signature phase

Canonical signed-statement encoding is not frozen by I0.

It must be frozen and covered by executable canonicalization fixtures before
any implementation is allowed to produce signatures.

## Checker remediation decisions

### Accepted checker changes

The checker will:

- validate exact normative clause IDs and their normalized hashes;
- verify cross-file model identifiers;
- parse public declarations through a positive type whitelist;
- require GLYPH_API and GLYPH_CALL on every exported function;
- exercise annotation-removal mutations;
- add C99 negative-array layout assertions;
- support --write and --verify modes;
- byte-compare a fresh canonical result in --verify mode;
- bind the result to an input_set_sha256 covering all reviewed source bytes.

### Modified external recommendations

The checker must not reject all C or C++ files under src because GLYPH already
contains unrelated compiled implementations.

Instead, pre-freeze implementation absence is checked by proving that:

- no definition of any public glyph_*_v1 ABI function exists;
- no embedded API shared-library target exists;
- no CMake target claims GLYPH_C_ABI_V1 implementation.

The tracked result does not contain its own enclosing Git commit SHA because
that would create a cyclic commit identity dependency.

The result is instead bound to input_set_sha256.

The external review package binds that input set to the reviewed Git commit.

## Required commit sequence

1. normative ABI and trust-model remediation;
2. executable checker remediation;
3. regenerated DRAFT_NOT_FROZEN result;
4. clean-checkout three-layer verification;
5. second deterministic external review package;
6. second pre-freeze review;
7. I0 freeze only after FREEZE_READY.
