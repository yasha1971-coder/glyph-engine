# GLYPH_OPERATOR_CORPUS_MANIFEST_V1

Status: executable implementation gate
Version: 1
Date: 2026-07-13
Operator obligation: O1

## Purpose

Define deterministic conversion of a filesystem tree into a committed,
binary-safe GLYPH operator corpus snapshot.

O1 does not build the FM runtime yet.

It establishes the stable mapping:

    filesystem file
    ->
    raw relative path identity
    ->
    document ID
    ->
    committed document bytes
    ->
    ordered corpus identity

## Accepted source objects

V1 accepts regular files and directories.

V1 rejects:

- symbolic links;
- sockets;
- devices;
- FIFOs;
- special filesystem objects;
- paths escaping the source root.

Empty regular files are valid documents.

## Path representation

The authoritative path field is:

    relative_path_bytes_hex

It is lowercase hexadecimal encoding of raw relative pathname bytes.

Display text is non-authoritative.

The authoritative path must be:

- relative;
- non-empty;
- free of NUL;
- free of empty components;
- free of `.` components;
- free of `..` components.

## Ordering

Documents are sorted lexicographically by raw relative pathname bytes.

Document IDs are assigned after sorting:

    0 .. document_count - 1

Filesystem enumeration order is never authoritative.

## Snapshot layout

A complete O1 snapshot contains:

    source_manifest_v1.json
    BUILD_COMPLETE_V1.json
    documents/doc_00000000.bin
    documents/doc_00000001.bin
    ...

Source filenames are represented in the manifest.

Committed document payloads use canonical numeric filenames.

## Registered extension directory

The O1 verifier reserves exactly one optional top-level extension:

    runtime_index_v1/

This directory is populated by O2.

O1 verifies that the extension, when present, is a real directory and not a
symbolic link.

O1 does not verify its internal runtime-index contents; that responsibility
belongs to O2.

Any unknown top-level file or directory remains forbidden.

## Source stability

For every file, the builder:

1. performs `lstat`;
2. opens without following a symbolic link;
3. performs `fstat`;
4. copies and hashes the file;
5. rereads and rehashes the open file descriptor;
6. performs final `fstat`;
7. performs final `lstat`;
8. requires identity and metadata stability;
9. requires both content hashes and lengths to agree.

At minimum the stable identity includes:

- device;
- inode;
- file type;
- byte length;
- modification time in nanoseconds.

Any instability aborts the complete build.

## Runtime corpus identity

The runtime corpus identity preimage is:

    ASCII("GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1")
    0x00
    document_count_u64_be

For each document in doc_id order:

    doc_id_u64_be
    byte_length_u64_be
    document_sha256_raw_32_bytes

## Source manifest identity

The source manifest identity preimage is:

    ASCII("GLYPH_OPERATOR_CORPUS_MANIFEST_V1")
    0x00
    document_count_u64_be

For each document in doc_id order:

    doc_id_u64_be
    relative_path_length_u64_be
    relative_path_raw_bytes
    byte_length_u64_be
    document_sha256_raw_32_bytes

Renaming a source file changes `source_manifest_id`.

Renaming alone does not change `corpus_id` when byte order and document
contents remain unchanged.

## Atomic publication

Construction occurs in a temporary sibling directory.

The final output path must not already exist.

The final directory is published only after:

- all document snapshots exist;
- all document hashes are verified;
- source discovery remains stable;
- canonical manifest is written;
- complete marker is written;
- snapshot verification succeeds.

A failed build must not publish the final output directory.

## Complete marker

`BUILD_COMPLETE_V1.json` binds:

- manifest SHA256;
- corpus ID;
- source manifest ID;
- document count.

A directory without a valid complete marker is not a complete operator corpus.

## Verification

Verification is independent of the original source directory.

It must reject:

- noncanonical JSON;
- missing files;
- undeclared files;
- symlinks;
- unsafe raw paths;
- incorrect document order;
- incorrect document IDs;
- document hash mismatch;
- document length mismatch;
- manifest identity mismatch;
- complete-marker mismatch.

## Completion criterion

O1 passes only when deterministic builds from equivalent filesystem trees
produce byte-identical manifests and all mutation fixtures are rejected.
