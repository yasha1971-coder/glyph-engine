# RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1

Status: measured local portable bundle verification  
Date: 2026-06-27

## Purpose

Validate a portable bounded evidence bundle for compressed RLBWT runtime.

Bundle version:

    RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1

Artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

Runtime profile:

    RLBWT_FULL_RUNTIME_PROFILE_V1

## Bundle contents

The bundle packages:

- `bundle_manifest_v1.json`
- `README_REPLAY.md`
- `corpus.bin`
- `evidence.json`
- `runtime/bwt.rlbwt`
- `runtime/bwt.rlbwt.rank`
- `runtime/locate_core_s128.bin`
- `runtime/manifest.json`
- `runtime/rlbwt_full_runtime_manifest_v1.json`

## Tools

Create bundle:

    tools/make_rlbwt_bounded_evidence_bundle_v1.py

Verify bundle:

    tools/verify_rlbwt_bounded_evidence_bundle_v1.py

## Verified tiny fixture

The bundle was created from the tiny RLBWT bounded evidence fixture.

Replay result:

- query: `the`
- FM interval: [65, 68]
- match_count: 3
- max_offsets: 2
- returned_count: 2
- bounded: true
- offsets: [43, 55]
- byte_check: true
- replay: PASS

## Meaning

This converts RLBWT bounded evidence from a repo-local artifact into a portable replay package.

A reviewer can receive the bundle directory and replay verification from the GLYPH repository root.

## Current boundary

The bundle is directory-based, not a tar/zip archive.

Still missing:

- JSON schema
- bundle spec document
- tarball packaging
- path-independent public release example
- integration with existing Evidence Bundle V1 family
