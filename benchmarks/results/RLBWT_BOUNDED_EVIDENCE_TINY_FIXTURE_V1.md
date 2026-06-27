# RLBWT_BOUNDED_EVIDENCE_TINY_FIXTURE_V1

Status: local reproducible tiny fixture  
Date: 2026-06-27

## Purpose

Validate a tiny reproducible RLBWT bounded evidence fixture.

Runner:

    tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh

The fixture creates a small corpus, builds a GLYPH index, exports RLBWT full runtime, creates bounded evidence, and replay-verifies the artifact.

## Corpus

Generated corpus:

    alpha beta gamma
    the quick brown fox
    delta the epsilon
    the final line

Query:

    the

max_offsets:

    2

Expected:

- match_count: 3
- returned_count: 2
- bounded: true
- byte_check: true
- replay verifier: PASS

## Meaning

This fixture turns RLBWT bounded evidence replay from a Pizza50 `/tmp` demonstration into a small reproducible repo workflow.

It is suitable as the next candidate for inclusion in `./verify.sh`.

## Repair validation

Repair commit:

- `1d7d180 fix: repair RLBWT bounded evidence tiny fixture runner`

Validation:

- runner executed successfully after repair
- bounded evidence artifact created
- replay verifier passed
- tiny fixture PASS

