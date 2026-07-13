# GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1

Status: executable implementation gate
Version: 1
Date: 2026-07-14
Operator obligation: O4

## Purpose

Define a deterministic, self-contained, portable evidence bundle produced from:

- an O1 committed filesystem snapshot;
- O2 compiled binary-safe indexes;
- an O3 exact binary query result.

The bundle must replay without:

- the original source directory;
- the GLYPH repository;
- Python project modules;
- network access;
- undeclared data.

## Bundle contents

A complete bundle contains:

    bundle_manifest_v1.json
    BUNDLE_COMPLETE_V1.json
    replay.py

    artifact/query_result_v1.json
    query/query.bin

    source/source_manifest_v1.json
    source/BUILD_COMPLETE_V1.json
    source/documents/doc_XXXXXXXX.bin

    runtime/runtime_manifest_v1.json
    runtime/RUNTIME_BUILD_COMPLETE_V1.json
    runtime/documents/doc_XXXXXXXX/sa.bin
    runtime/documents/doc_XXXXXXXX/bwt.bin
    runtime/documents/doc_XXXXXXXX/fm.bin

    bin/build_sa_binary_v1
    bin/build_bwt_binary_v1
    bin/build_fm_binary_v1
    bin/query_fm_binary_v1
    bin/query_fm_locate_binary_v1

## Completeness

The bundle manifest declares every payload other than:

- the bundle manifest itself;
- the bundle complete marker.

Exact coverage is mandatory.

Missing files, undeclared files, duplicate paths, symbolic links, non-regular
payloads, unsafe paths, and unexpected executable permissions are rejected.

## Integrity

Every declared payload binds:

- canonical relative path;
- role;
- byte length;
- SHA256;
- executable flag.

`bundle_root_sha256` commits to the ordered set of payload records.

`BUNDLE_COMPLETE_V1.json` binds:

- bundle manifest SHA256;
- bundle root SHA256;
- query result ID;
- runtime index ID;
- corpus ID;
- file count.

## Included compiled binaries

The bundle contains the exact runtime builder binaries committed by O2 and the
exact query binaries committed by O3.

The standalone replay verifier checks their byte lengths, SHA256 values, and
executable permissions.

## Standalone replay

`replay.py` uses only:

- the Python standard library;
- compiled binaries included in `bin/`;
- payloads included in the bundle.

Replay performs:

1. exact bundle coverage verification;
2. payload hash verification;
3. O1 source-manifest and corpus-identity verification;
4. O2 runtime-manifest and runtime-index-identity verification;
5. query identity verification;
6. compiled FM count and locate for every document;
7. count/locate agreement checks;
8. independent byte checks for every returned coordinate;
9. reconstruction of the complete O3 result;
10. exact comparison with `artifact/query_result_v1.json`.

## Portability boundary

V1 is portable across compatible Linux environments for the included compiled
executables.

V1 is not:

- a virtual-machine image;
- a container image;
- architecture-independent executable code;
- a guarantee of compatibility with every libc or kernel.

These limitations do not create hidden data dependencies.

## Determinism

Equivalent O1 snapshots, equivalent O2 indexes, the same query, the same limit,
and the same compiled binaries must produce byte-identical bundle payloads and
manifests.

The bundle excludes:

- absolute source paths;
- temporary build paths;
- timestamps;
- host names;
- elapsed time;
- random identifiers.

## Atomic publication

Construction occurs in a temporary sibling directory.

The final output directory must not already exist.

The final bundle becomes visible only after standalone replay succeeds.

A failed build must not publish the final output directory.

## Completion criterion

O4 passes only when:

- deterministic bundles are byte-identical;
- copied-bundle replay succeeds outside the repository;
- original source and corpus directories can be removed before replay;
- repository and network dependencies are absent;
- all declared mutations are rejected.
