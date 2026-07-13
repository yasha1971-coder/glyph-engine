# GLYPH_OPERATOR_RUNTIME_INDEX_V1

Status: executable implementation gate
Version: 1
Date: 2026-07-13
Operator obligation: O2

## Purpose

Build deterministic binary-safe C++ runtime indexes exclusively from the
committed O1 operator snapshot.

O2 establishes:

    committed document bytes
    ->
    private verified build input
    ->
    deterministic SA/BWT/FM
    ->
    committed runtime index

The original filesystem source directory is never used by O2.

## Dependency

O2 requires a valid:

    GLYPH_OPERATOR_CORPUS_MANIFEST_V1

The O1 snapshot must independently verify before index construction begins.

## Runtime implementation

O2 invokes the verified compiled binaries:

- `build_sa_binary_v1`
- `build_bwt_binary_v1`
- `build_fm_binary_v1`

The runtime profile is:

    GLYPH_BINARY_RUNTIME_V1

## Runtime semantics

The index supports:

- every source byte from `0x00` through `0xFF`;
- logical virtual sentinel `256`;
- 257-symbol alphabet;
- one independent index per document;
- empty documents;
- duplicate documents;
- deterministic document-local coordinates.

## Index topology

The topology is:

    one_independent_index_per_document

No physical concatenation of documents is permitted.

The resulting directory is:

    runtime_index_v1/

## Runtime index layout

A complete runtime index contains:

    runtime_index_v1/
        runtime_manifest_v1.json
        RUNTIME_BUILD_COMPLETE_V1.json
        documents/
            doc_00000000/
                sa.bin
                bwt.bin
                fm.bin
            doc_00000001/
                sa.bin
                bwt.bin
                fm.bin
            ...

No undeclared payload is permitted.

## Source isolation

For each committed document O2 must:

1. verify the O1 snapshot;
2. open the committed numeric document payload without following symlinks;
3. copy it into a private temporary build input;
4. verify byte length and SHA256 against the O1 manifest;
5. build SA/BWT/FM only from the private copy;
6. reread the committed payload after runtime construction;
7. require the committed payload to remain identical;
8. discard the private build input before publication.

This prevents runtime construction from consuming a changing source pathname.

## Runtime binary commitments

The runtime manifest binds the exact builder binaries used:

- binary name;
- byte size;
- SHA256.

The manifest does not bind an absolute build path.

## Per-document commitments

Each document record binds:

- doc_id;
- committed source snapshot path;
- source byte length;
- source SHA256;
- canonical runtime index directory;
- SA format, size, path, and SHA256;
- BWT format, size, path, and SHA256;
- FM format, size, path, and SHA256.

## Formats

The committed artifact format labels are:

    GLYPH_SA_BINARY_V1
    GLYPH_BWT_BINARY_V1
    GLYPH_FM_BINARY_V1

## Runtime index identity

The runtime index identity binds:

- O1 source manifest SHA256;
- corpus ID;
- source manifest ID;
- runtime topology;
- logical sentinel;
- alphabet size;
- FM checkpoint step;
- runtime binary commitments;
- ordered per-document source commitments;
- ordered SA/BWT/FM commitments.

The identity is independent of:

- absolute corpus path;
- temporary construction path;
- wall-clock time;
- build duration;
- host display name.

## Atomic publication

Runtime construction occurs in a temporary directory outside the committed
corpus directory.

The final path:

    runtime_index_v1/

becomes visible only after:

- every private input is verified;
- every SA/BWT/FM artifact is produced;
- all artifact hashes are calculated;
- the runtime manifest is written;
- the runtime complete marker is written;
- structural verification passes;
- deterministic rebuild verification passes when requested.

Publication uses an atomic rename on the same filesystem.

A failed or interrupted build must not publish `runtime_index_v1/`.

## Existing index policy

V1 never overwrites an existing `runtime_index_v1/`.

Rebuilding requires explicit removal by a future separate policy.

## Verification modes

Structural verification checks:

- O1 snapshot validity;
- exact runtime directory coverage;
- canonical JSON;
- source-manifest binding;
- corpus identity binding;
- runtime-index identity;
- per-file byte sizes and SHA256;
- complete-marker binding;
- absence of symlinks and undeclared files.

Rebuild verification additionally:

- loads the current verified C++ builder binaries;
- checks their commitments;
- rebuilds every document in a temporary directory;
- requires byte-identical SA/BWT/FM commitments.

## Completion criterion

O2 passes only when:

- equivalent O1 snapshots produce byte-identical runtime manifests;
- equivalent builds produce byte-identical runtime payloads;
- empty and arbitrary-byte documents build successfully;
- source mutation is rejected;
- interrupted construction is not published;
- index mutations are rejected;
- deterministic rebuild verification passes.
