# GLYPH_EMBEDDED_THREAT_MODEL_V1

Status: normative draft; implementation blocked
Version: 1
Date: 2026-07-14
Phase: I0 contract freeze

## Purpose

Define the security and reliability boundary for the first embeddable GLYPH
runtime.

This specification governs the future public C ABI, immutable mmap loader,
concurrent read-only query path, resource-failure behavior, and signed evidence
statement.

It does not claim that these capabilities are implemented.

## Verified dependency

The embedded work begins from:

- commit:
  `8e8019b42a9c23e8d0b9d1c81ab815503952f877`
- tag:
  `glyph-operator-path-v1-verified`

Verified lower-layer closure:

    P1-P12
    ->
    R0-R6
    ->
    O1-O6
    ->
    VERIFY OK

That closure remains valid.

It does not automatically prove correctness or safety of a new public ABI,
mmap parser, concurrency layer, timeout mechanism, or signature verifier.

## Protected properties

The embedded runtime must preserve:

1. exact binary query semantics;
2. canonical match count;
3. canonical document coordinates;
4. document-local boundary semantics;
5. corpus and runtime-index identity;
6. bounded locate completeness semantics;
7. deterministic evidence identity;
8. process availability under rejected input where the operating system permits
   controlled failure;
9. absence of unlabeled partial success.

## Hostile inputs

The implementation must treat the following as hostile:

- every index file before structural validation;
- every manifest and metadata field;
- every declared length, count, offset, alignment, and section boundary;
- every query byte and query length;
- every caller-provided result capacity;
- every source-path byte stored in the manifest;
- truncated files;
- oversized declared fields;
- internally inconsistent but correctly hashed files;
- unsupported versions;
- duplicate and overlapping sections;
- files changed during open;
- concurrent API call ordering.

A valid SHA-256 value does not make a file structurally safe.

The parser must be hardened as though integrity verification did not exist.

## Trusted assumptions

Embedded Runtime V1 assumes:

- a compatible supported operating system and CPU architecture;
- valid readable and writable caller memory for all pointers supplied to the
  C ABI;
- the caller obeys successful-close lifetime rules;
- the published index filesystem follows
  `GLYPH_TRUSTED_IMMUTABLE_LOCAL_FILESYSTEM_V1`;
- the process is not controlled by a privileged attacker;
- executable code and loaded shared libraries are trusted.

The C ABI cannot safely recover from an invalid caller pointer that refers to
unmapped or inaccessible memory.

Such caller-memory violations are outside the hostile-input guarantee.

## Filesystem attacker boundary

Before successful open, index bytes are untrusted.

During open, the runtime must verify:

- regular-file type;
- expected manifest coverage;
- actual file sizes;
- SHA-256 commitments;
- structural format validity;
- stable file identity during verification.

After successful open, the V1 trust model requires that published inodes are
not modified or truncated.

A privileged local actor who mutates an already-open inode is outside the V1
trust model.

## Query boundary

All 256 byte values are valid query data.

The query pointer is borrowed only for the duration of the call.

An empty query is rejected.

Query processing must not:

- interpret query bytes as text;
- search for a terminator;
- read beyond the declared query length;
- allocate memory proportional to total match count;
- expose a terminal sentinel as a source byte.

## Result boundary

The caller owns all coordinate buffers.

The runtime must return separately:

- total match count;
- returned coordinate count;
- completeness.

A successful bounded locate may be incomplete only because of the explicit
caller-provided bound.

Timeout, memory failure, verification failure, or internal failure must not be
reported as successful bounded evidence.

## Concurrency boundary

A successfully opened handle is intended to support concurrent read-only
operations.

The shared handle state must be immutable after open.

Forbidden shared state includes:

- mutable query cursors;
- lazy mutable lookup tables;
- shared scratch buffers;
- process-global mutable error records;
- unsynchronized cached results.

Closing a handle while operations are active must return `GLYPH_E_BUSY`.

After a close attempt begins, no new operation may begin on that handle.

The caller is responsible for establishing this close barrier. The raw C
pointer does not provide automatic lifetime protection against a caller that
begins a new operation concurrently with successful destruction.

## Resource boundary

The query plane is an in-process library and cannot guarantee survival from
every operating-system OOM policy.

The runtime must still:

- reject declared sizes before unsafe allocation;
- use checked arithmetic;
- return `GLYPH_E_NOMEM` for catchable allocation failure;
- enforce configured logical limits;
- avoid output allocation proportional to match count.

Heavy index construction belongs in an isolated build worker with external
memory and time enforcement.

## Availability non-claims

Until executable gates exist, GLYPH does not claim:

- hostile-index safety;
- OOM-proof in-process execution;
- timeout enforcement;
- crash-safe power-loss recovery;
- thread safety;
- lock freedom;
- denial-of-service resistance;
- side-channel resistance;
- compatibility with unsupported platforms.

## Out of scope for V1

The following are outside the first embedded-runtime threat model:

- malicious kernel or hypervisor;
- malicious root or equivalent local administrator;
- arbitrary invalid caller pointers;
- speculative-execution attacks;
- physical memory attacks;
- continuous post-open Merkle verification;
- network transport security;
- key issuance and organizational identity proof.

## Required future gates

The threat model is not considered implemented until at least:

- strict C ABI parity;
- hostile mmap parser checks;
- ASan;
- UBSan;
- TSan concurrent-reader testing;
- malformed-index fuzzing;
- query API fuzzing;
- resource-failure injection;
- clean-checkout replay;
- deterministic result verification.
