# GLYPH_OPERATOR_PATH_V1

Status: implementation contract
Version: 1
Date: 2026-07-13

## Purpose

Define the first real-world operator workflow over the verified
GLYPH binary-safe C++ runtime.

The operator path must transform an ordinary filesystem corpus into:

- a deterministic ordered corpus;
- stable document identities;
- binary-safe exact queries;
- canonical source coordinates;
- a self-contained evidence bundle;
- independently replayable results.

The verified C++ runtime remains authoritative.

This layer orchestrates the runtime. It does not replace or reimplement it.

## Required operator workflow

The intended public workflow is:

    glyphctl build <source-directory> --out <corpus-directory>

    glyphctl query <corpus-directory> \
        --query-file <query.bin> \
        --max-offsets <N> \
        --bundle <evidence-directory>

    glyphctl replay <evidence-directory>

    glyphctl inspect <corpus-directory-or-bundle>

## Architecture boundary

The operator layer may:

- discover filesystem documents;
- create a canonical source manifest;
- assign stable document IDs;
- invoke verified C++ runtime binaries;
- collect count and locate results;
- create runtime evidence artifacts;
- create self-contained bundles;
- emit human-readable and machine-readable output.

The operator layer must not:

- implement an independent search algorithm;
- use grep, strstr, text decoding, or tokenization as the result oracle;
- silently skip unreadable files;
- trust filename display text as authoritative identity;
- accept partial index builds as complete;
- weaken P1-P12 or R0-R6 invariants.

## Source document domain

A source corpus is an explicitly selected filesystem tree.

V1 accepts only regular files.

V1 rejects by default:

- symbolic links;
- directories encountered as document payloads;
- sockets;
- devices;
- FIFOs;
- unresolved paths;
- paths escaping the selected source root.

Empty regular files are valid documents.

File contents may contain every byte:

    0x00 through 0xFF

## Canonical path identity

Filesystem paths are not assumed to be valid UTF-8.

The authoritative source-path representation is:

    relative_path_bytes_hex

The preimage is the raw relative pathname byte sequence as represented by the
host filesystem.

The separator between path components is the byte:

    0x2F

The following are forbidden:

- absolute paths;
- empty path components;
- "." components;
- ".." components;
- NUL bytes;
- paths outside the selected root.

A non-authoritative display path may be included for user interfaces.

Search, replay, corpus identity, and manifest verification must never depend on
the display path.

## Canonical document ordering

Documents are sorted lexicographically by:

    relative_path_bytes

Document IDs are assigned after sorting:

    0, 1, 2, ... document_count - 1

Traversal order returned by the operating system is not authoritative.

Duplicate file contents remain separate documents when their path identities
differ.

Empty files retain their assigned document IDs.

## Source manifest

The source manifest version is:

    GLYPH_OPERATOR_CORPUS_MANIFEST_V1

The manifest contains:

- manifest version;
- document count;
- ordered document records;
- runtime corpus ID;
- source manifest ID;
- construction status.

Each document record contains:

- doc_id;
- relative_path_bytes_hex;
- optional display_path;
- byte_length;
- SHA256;
- document type equal to regular_file.

## Identity separation

The existing runtime corpus identity remains the authoritative ordered
byte-corpus identity.

It binds:

- document order;
- document lengths;
- document SHA256 values.

The operator source manifest identity additionally binds filesystem path
identity.

Therefore GLYPH records two separate identities:

    corpus_id
    source_manifest_id

Changing a path without changing file bytes preserves `corpus_id` but changes
`source_manifest_id`.

Changing document order, length, or contents changes both identities.

## Source manifest identity

The source manifest identity preimage is:

    ASCII("GLYPH_OPERATOR_CORPUS_MANIFEST_V1")
    0x00
    document_count_u64_be

For every document in doc_id order:

    doc_id_u64_be
    relative_path_length_u64_be
    relative_path_raw_bytes
    byte_length_u64_be
    document_sha256_raw_32_bytes

The resulting identity is SHA256 of the complete preimage.

## Stable-file requirement

A file must not silently change while it is being committed.

For each source document the builder must verify stability across the read:

- initial metadata observation;
- content read and SHA256 calculation;
- final metadata observation.

At minimum V1 compares:

- device ID;
- inode;
- byte length;
- nanosecond modification time when available.

If stability cannot be established, the build fails.

A changed file must never produce a successful corpus manifest.

## Atomic construction

Corpus construction must occur in a temporary sibling directory.

The final corpus directory becomes visible only after:

- all source files were processed;
- all hashes were calculated;
- all runtime indexes were built;
- all runtime commitments were validated;
- the manifest was written;
- the build-complete marker was written.

Final publication uses an atomic rename when supported.

An interrupted build must not appear as a complete corpus.

Existing output directories are not overwritten unless an explicit future
policy permits it.

## Query transport

The authoritative query is a non-empty byte sequence.

Supported V1 inputs are mutually exclusive:

    --query-file <path>
    --query-hex <canonical-lowercase-hex>

`--query-file` is the preferred real-world interface.

Shell strings and display text are never authoritative query transports.

The query identity contains:

- query_hex;
- query_length_bytes;
- query_sha256.

Empty queries are rejected.

## Query results

Operator query results use:

    GLYPH_OPERATOR_QUERY_RESULT_V1

Required fields include:

- corpus_id;
- source_manifest_id;
- query identity;
- match_count;
- canonical coordinates;
- returned_count;
- bounded;
- offsets_complete;
- byte_check;
- runtime profile;
- evidence bundle path when requested.

Coordinates are:

    (document_id, document_offset)

The operator may add source-path display metadata, but the canonical coordinate
remains numeric.

## Count and locate authority

Count and locate results must come from the verified compiled runtime:

- `query_fm_binary_v1`;
- `query_fm_locate_binary_v1`.

The operator layer may aggregate per-document results.

It may not invent, infer, or repair runtime results.

Every returned coordinate is byte-checked against its committed document.

## Evidence bundle

When `--bundle` is requested, the produced bundle must include:

- source manifest;
- runtime evidence artifact;
- ordered source documents;
- required runtime binaries;
- replay module;
- replay entrypoint;
- exact manifest coverage;
- payload hashes;
- bundle root.

Replay must not require:

- the original source directory;
- the GLYPH repository;
- network access.

## Exit status contract

V1 uses:

    0  success
    2  invalid command or invalid user input
    3  source discovery or source stability failure
    4  runtime construction failure
    5  runtime query failure
    6  evidence or bundle verification failure
    7  internal invariant failure

Failures must also emit a machine-readable JSON error object.

## Output contract

Normal command output supports canonical JSON.

Progress and diagnostic messages are written to stderr.

JSON result data is written to stdout.

A successful JSON object must include:

    "ok": true

A failed JSON object must include:

    "ok": false
    "error_code"
    "error"

Partial success is forbidden for V1.

## Performance observations

The operator path records, without turning them into correctness claims:

- source document count;
- source byte count;
- discovery duration;
- hashing duration;
- index-build duration;
- query duration;
- evidence-build duration;
- per-document runtime size;
- total runtime size.

Performance measurements must not alter artifact identity.

## Required fixtures

The operator gate must include:

1. ASCII files.
2. Embedded NUL bytes.
3. Byte `0xFF`.
4. Invalid UTF-8 contents.
5. Invalid UTF-8 filename bytes.
6. Empty file.
7. Duplicate file contents under different paths.
8. Nested directories.
9. Cross-document-only query.
10. Symlink rejection.
11. File mutation during build.
12. Interrupted-build residue.
13. Query supplied through a binary file.
14. Zero-match query.
15. Bounded multi-document locate.
16. Copied evidence replay outside the repository.

## Completion criterion

`GLYPH_OPERATOR_PATH_V1` is complete only when a clean-checkout verifier proves:

- deterministic source discovery;
- deterministic doc_id assignment;
- stable source-manifest identity;
- binary-safe query-file transport;
- exact runtime count and locate;
- correct source-path mapping;
- atomic corpus construction;
- source mutation rejection;
- symlink and traversal rejection;
- self-contained evidence replay;
- no repository or network dependency.

## Non-claims

V1 does not claim:

- support for remote object stores;
- support for live-changing directories;
- support for special filesystem objects;
- incremental index updates;
- distributed index construction;
- a single unified FM index across all documents;
- cross-platform binary portability;
- production service availability.

Those require separate evidence and measurement.
