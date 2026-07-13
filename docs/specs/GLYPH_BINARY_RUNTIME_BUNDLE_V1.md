# GLYPH_BINARY_RUNTIME_BUNDLE_V1

Status: implementation gate  
Version: 1  
Date: 2026-07-13

## Purpose

Define a self-contained portable bundle for evidence produced by the compiled
GLYPH binary-safe C++ runtime.

The bundle must contain every project-controlled component required to replay
the evidence without access to the GLYPH repository or original source paths.

## Mandatory payload

A bundle contains:

- `evidence.json`;
- all ordered source documents;
- the evidence replay module;
- an autonomous replay entrypoint;
- the exact C++ runtime binaries used for replay;
- `bundle_manifest_v1.json`.

## Runtime binaries

Required bundled binaries:

- `build_sa_binary_v1`;
- `build_bwt_binary_v1`;
- `build_fm_binary_v1`;
- `query_fm_locate_binary_v1`.

Replay must use these bundled binaries rather than binaries from the repository
or the host PATH.

## Manifest

The manifest records every payload file with:

- canonical relative path;
- semantic role;
- byte size;
- SHA256;
- executable flag.

The manifest must exactly cover the bundle payload. Undeclared files, missing
files, symlinks, absolute paths, and parent traversal are forbidden.

## Bundle root

`bundle_root_sha256` is SHA256 of the canonical JSON serialization of the
sorted payload file records.

The manifest itself is excluded from this non-recursive root.

## Replay

The replay entrypoint must:

1. validate the manifest;
2. verify exact payload coverage;
3. reject symlinks and unsafe paths;
4. verify the size and SHA256 of every payload;
5. verify `bundle_root_sha256`;
6. load the bundled documents in canonical document-ID order;
7. load the bundled evidence artifact;
8. use the bundled replay module;
9. use only the bundled C++ binaries;
10. rebuild deterministic runtime indexes;
11. compare index commitments;
12. rerun count and locate;
13. verify every result and byte coordinate.

## External dependencies

The bundle has no external data dependency and no GLYPH repository dependency.

The V1 portability boundary still assumes:

- a compatible Linux userspace;
- Python 3 standard library;
- shared system libraries required by the bundled C++ executables.

This is not a hermetic virtual machine or container image.

## Completion criterion

The bundle must replay after being copied outside the repository and after the
original source-document paths have been removed.

Every mutation of a declared payload or manifest invariant must be rejected.
