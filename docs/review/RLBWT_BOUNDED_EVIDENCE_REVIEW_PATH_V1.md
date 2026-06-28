# RLBWT_BOUNDED_EVIDENCE_REVIEW_PATH_V1

Status: draft  
Date: 2026-06-27

## Purpose

This document gives a short review path for the experimental GLYPH RLBWT bounded evidence subsystem.

It is intended for an external technical reviewer who wants to verify the current compressed bounded evidence chain from the repository root.

## What this path verifies

The current review path verifies:

    tiny corpus
    -> sentinel-safe GLYPH index
    -> RLBWT full runtime
    -> bounded evidence artifact
    -> artifact replay verifier
    -> portable bundle creation
    -> bundle replay verifier
    -> byte_check

The tarball path additionally verifies:

    portable bundle directory
    -> deterministic tar.gz
    -> safe extract
    -> bundle replay verifier
    -> artifact replay verifier

## One-command verification

From the repository root:

    ./verify.sh

Expected high-level result:

    [tiny-fixture] PASS
    VERIFY OK

This runs the tiny RLBWT bounded evidence fixture and verifies both artifact replay and portable bundle replay.

## Tiny fixture

Runner:

    tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh

The fixture creates this source corpus:

    alpha beta gamma
    the quick brown fox
    delta the epsilon
    the final line

Query:

    the

Expected verified result:

- FM interval: [65, 68]
- match_count: 3
- max_offsets: 2
- returned_count: 2
- bounded: true
- offsets: [43, 55]
- byte_check: true
- artifact replay: PASS
- bundle replay: PASS

## Create a portable bundle tarball

After running the tiny fixture, create a tarball:

    python3 tools/make_rlbwt_bounded_evidence_bundle_tar_v1.py \
      --artifact examples/rlbwt-bounded-evidence-tiny/out/rlbwt_bounded_evidence_v1.json \
      --out-tar /tmp/glyph_rlbwt_bounded_evidence_bundle_v1.tar.gz \
      --force

Expected output includes:

- tar_version: RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1
- bundle_version: RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1
- artifact_version: RLBWT_BOUNDED_EVIDENCE_V1
- ok: true

## Verify the tarball

    python3 tools/verify_rlbwt_bounded_evidence_bundle_tar_v1.py \
      --tar /tmp/glyph_rlbwt_bounded_evidence_bundle_v1.tar.gz

Expected result:

- ok: true
- errors: []
- replay_result.ok: true
- byte_check: true inside nested artifact replay result

## Current technical state

Current technical state map:

    docs/review/GLYPH_CURRENT_TECHNICAL_STATE_V1.md

This document separates verified working paths, reference-only binary-safe work, known boundaries, explicit non-claims, and implementation triggers.

## Relevant specs

Artifact spec:

    docs/specs/RLBWT_BOUNDED_EVIDENCE_SPEC_V1.md

Bundle spec:

    docs/specs/RLBWT_BOUNDED_EVIDENCE_BUNDLE_SPEC_V1.md

## Relevant reports

Artifact replay:

    benchmarks/results/RLBWT_BOUNDED_EVIDENCE_V1_REPLAY.md

Tiny fixture:

    benchmarks/results/RLBWT_BOUNDED_EVIDENCE_TINY_FIXTURE_V1.md

Portable bundle:

    benchmarks/results/RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1.md

Tarball packaging:

    benchmarks/results/RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1.md

## Non-claims

This path does not claim:

- exhaustive locate when bounded=true
- semantic search
- ranking
- fuzzy matching
- token search
- legal proof by itself
- signed or notarized artifact authenticity

This path demonstrates only that the current repository can reproduce and replay a bounded exact evidence chain over compressed RLBWT runtime for the included tiny fixture.

## Current boundary

The tarball is not signed.

The fixture is intentionally tiny and correctness-focused.

Larger corpus replay exists in measured reports, but this review path is designed to be fast and reproducible.
