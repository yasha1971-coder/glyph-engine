# RLBWT_BOUNDED_EVIDENCE_V1_REPLAY

Status: measured local replay verification  
Date: 2026-06-27

## Purpose

Validate replay verification for RLBWT bounded evidence artifact.

Artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

Replay verifier:

    tools/verify_rlbwt_bounded_evidence_v1.py

## Replay checks

The verifier checks:

- artifact version
- runtime profile
- query hex
- query SHA256
- source corpus size/hash
- runtime file size/hash
- recomputed FM interval
- recomputed match_count
- recomputed bounded offsets
- byte check for returned offsets

## Test artifact

Artifact:

    /tmp/glyph_rlbwt_bounded_evidence_v1/pizza_the_max10_evidence.json

Query:

    the

max_offsets:

    10

Expected replay result:

- FM interval: [45130554, 45891956]
- match_count: 761402
- returned_count: 10
- bounded: true
- byte_check: true
- replay: PASS

## Meaning

RLBWT bounded evidence is now replay-verifiable.

This converts the path from artifact creation into a reproducible evidence chain:

    compressed runtime
    -> bounded server retrieval
    -> bounded evidence artifact
    -> replay verifier

## Current boundary

Still missing:

- portable bundle format
- stable JSON schema
- integration with existing Audit Artifact V0 / Evidence Case V1
- CI fixture using small corpus
- server protocol documentation
