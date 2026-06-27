# RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1

Status: measured local tarball packaging verification  
Date: 2026-06-27

## Purpose

Validate tar.gz packaging for portable RLBWT bounded evidence bundles.

Tar version:

    RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1

Contained bundle version:

    RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1

Contained artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

Runtime profile:

    RLBWT_FULL_RUNTIME_PROFILE_V1

## Tools

Create tarball:

    tools/make_rlbwt_bounded_evidence_bundle_tar_v1.py

Verify tarball:

    tools/verify_rlbwt_bounded_evidence_bundle_tar_v1.py

## Verification model

The tar verifier:

1. opens the tar.gz archive
2. rejects unsafe absolute/path-traversal entries
3. rejects hard links and symlinks
4. requires exactly one top-level bundle directory
5. extracts into a temporary directory
6. invokes the portable bundle verifier
7. delegates artifact semantics to the bounded evidence replay verifier

## Verified tiny fixture

The tarball was created from the tiny RLBWT bounded evidence fixture.

Replay result:

- query: `the`
- FM interval: [65, 68]
- match_count: 3
- max_offsets: 2
- returned_count: 2
- bounded: true
- offsets: [43, 55]
- byte_check: true
- bundle replay: PASS
- tar replay: PASS

## Meaning

The portable bounded evidence bundle can now be packaged as a single `.tar.gz` file and replay-verified after extraction.

This moves the RLBWT bounded evidence path closer to an externally shareable artifact.

## Current boundary

The tarball is not signed.

Still missing:

- JSON schema
- signature/checksum sidecar
- public release example
- integration with existing Evidence Bundle V1 family
- optional tarball verification inside `./verify.sh`
