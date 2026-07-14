# GLYPH_MMAP_TRUST_MODEL_V1

Status: normative draft; implementation blocked
Version: 1
Date: 2026-07-14
Trust profile: GLYPH_TRUSTED_IMMUTABLE_LOCAL_FILESYSTEM_V1

## Purpose

Define exactly what integrity claim may be made when GLYPH uses read-only mmap
for immutable runtime indexes.

## Selected V1 model

Embedded Runtime V1 verifies complete files at successful open and then relies
on an immutable local-filesystem publication discipline.

It does not cryptographically re-verify every page on every later access.

## Required publication model

A published runtime index is immutable.

Publication immutability begins before the index becomes available to open and
continues for the complete lifetime of every open handle.

A new version must be created under new temporary inodes, fully written,
verified, synchronized, and then published atomically.

A publisher must not:

- overwrite a published payload in place;
- truncate a published payload;
- append to a published payload;
- reuse a published inode for different bytes.

Replacing a directory entry with a new inode does not alter already-open file
descriptors or mappings.

The default V1 trust profile does not include an authorized writer that
mutates a published inode.

## Root directory anchoring

The index-directory path is resolved once to an opened directory descriptor.

All manifest and payload files are then opened relative to that descriptor.

The implementation must not repeatedly resolve the original path for each
payload.

Payload opens must:

- reject symlink payloads;
- remain beneath the opened index root;
- use descriptor-relative lookup;
- prevent path replacement from selecting a different runtime tree during the
  same open operation.

Integrity verification, structural validation, and mmap of one payload must
operate on the same opened file description.

## Required open sequence

For every declared runtime payload, open must conceptually perform:

1. open a file descriptor for read-only access;
2. reject symlink payloads and non-regular files;
3. obtain file identity and size through `fstat`;
4. enforce configured size, document, and address-space limits;
5. verify manifest coverage and reject impossible mapping sizes;
6. create the final read-only mapping from that opened file description;
7. compute full SHA-256 through the final mapped region;
8. compare the hash with the committed manifest value;
9. parse and structurally validate all headers, sections, bindings, and source
   paths through that same final mapped region;
10. re-check descriptor identity and supported metadata as a best-effort
    mutation check;
11. expose the handle only after all payloads pass.

No query may dereference a mapped field before structural validation succeeds.

Failure at any step must prevent handle publication.

## Required structural checks

Before a mapped field is dereferenced as a structure, the implementation must
validate:

- magic and format version;
- fixed on-disk endianness;
- total file size;
- section count;
- section offset;
- section size;
- offset plus size using checked arithmetic;
- multiplication using checked arithmetic;
- section alignment;
- section overlap;
- element width;
- declared count versus section capacity;
- non-zero divisors and block sizes;
- cross-file identity bindings;
- terminal-sentinel invariants;
- document-count and source-manifest consistency.

Every committed source path must also be validated as:

- non-empty;
- relative;
- free of byte `0x00`;
- encoded with byte `0x2F` as its component separator;
- free of leading and trailing separators;
- free of empty components;
- free of `.` and `..` components;
- unique as a raw byte sequence within the index.

Invalid UTF-8 is permitted.

A correct hash does not waive any structural or source-path check.

## mmap properties

Mappings must be:

- read-only;
- private or otherwise non-writable by the GLYPH process;
- retained by the handle for the complete handle lifetime.

The C ABI must not expose a pointer into a mapped region.

## Permitted integrity claim

After successful open, GLYPH may state:

> Under `GLYPH_TRUSTED_IMMUTABLE_LOCAL_FILESYSTEM_V1`, the bytes observed
> through the final runtime mappings during open matched their committed
> SHA-256 values and passed structural validation before handle publication.

This claim depends on the publication inode remaining immutable from before
open through the complete handle lifetime.

## Forbidden integrity claim

GLYPH must not state:

> Every byte served from mmap is cryptographically verified at the time of each
> query.

## Post-open mutation

Mutation or truncation of an already-open published inode violates the V1
trust model.

An ordinary read-only private mapping does not cryptographically authenticate
every later page access and does not protect against a writer that is allowed
to mutate the backing inode.

On operating systems where truncating an active mapping can produce `SIGBUS`,
the V1 runtime does not claim that such a post-open trust-model violation is
recoverable in-process.

Deployments requiring protection from hostile local mutation need a stronger
profile, such as:

- filesystem immutability enforcement;
- fs-verity;
- dm-verity;
- segmented or Merkle-verified runtime formats;
- process isolation.

These are not part of the default V1 profile.

## Replacement race

Opening, hashing, validating, and mapping must use the same opened file
description.

Path lookup must not be repeated between integrity verification and mmap in a
way that could select a different inode.

## File change during open

The implementation must reject mutation detectable during open.

Before mapping and after mapped hashing and validation, it must compare
relevant identity and metadata obtained from the same opened descriptor,
including:

- device identity where available;
- inode identity where available;
- byte size;
- modification metadata supported by the platform.

A changed value aborts open.

This comparison is a best-effort check.

It does not detect every size-preserving rewrite, including a rewrite for
which modification metadata is restored.

Mapping, hashing, and validation through one final mapping prevents parser and
query code from intentionally using two different file paths or file
descriptions, but it does not make mutable local storage cryptographically
immutable.

The publication discipline is therefore a required trust assumption, not a
property inferred from metadata comparison.

## Platform scope

The first conformant implementation target is expected to be a supported
64-bit Linux environment.

Other operating systems are unverified until they have platform-specific
open, mapping, mutation, and failure gates.

## Future stronger format

A future format may use independently authenticated segments or a Merkle tree.

That would be a new index-format and trust-profile version.

It must not silently change the claim semantics of this V1 profile.
