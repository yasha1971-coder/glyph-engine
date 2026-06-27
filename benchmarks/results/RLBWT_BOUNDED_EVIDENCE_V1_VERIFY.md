# RLBWT_BOUNDED_EVIDENCE_V1_VERIFY

Status: measured local verification  
Date: 2026-06-27

## Purpose

Validate bounded evidence artifact creation from persistent C++ RLBWT server output.

Tool:

    tools/make_rlbwt_bounded_evidence_v1.py

Runtime profile:

    RLBWT_FULL_RUNTIME_PROFILE_V1

Server protocol:

    query_hex<TAB>max_offsets

Evidence artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

## Test case

Corpus:

    Pizza & Chili English 50MB prefix

Runtime:

    /tmp/glyph_rlbwt_full_runtime_v1/pizza50

Source corpus:

    examples/public-evidence-demo/work/pizza_english_50mb/corpus.bin

Query:

    the

max_offsets:

    10

## Result

The bounded evidence artifact was created successfully.

Observed retrieval:

- FM interval: [45130554, 45891956]
- match_count: 761402
- max_offsets: 10
- returned_count: 10
- bounded: true
- byte_check: true

Returned offsets:

- 786213
- 4736688
- 5595028
- 7801431
- 8761711
- 19383215
- 20264073
- 46448252
- 46448984
- 47354405

## Meaning

This verifies the first bounded evidence path over compressed RLBWT runtime.

The artifact contains:

- query text
- query hex
- query hash
- source corpus hash
- runtime file hashes
- FM interval
- full match_count
- max_offsets
- returned bounded offsets
- byte checks for returned offsets

This is not exhaustive locate.

It is bounded evidence:

    full exact count
    + deterministic bounded offsets
    + byte verification for returned offsets

## Strategic meaning

GLYPH now has a practical compressed evidence path for high-count queries.

For a query with 761402 matches, the system can produce an evidence artifact with full count and 10 verified offsets without exhaustive offset recovery.

This is the first bridge from RLBWT runtime work back into the audit/evidence chain.

## Current boundary

Still missing:

- replay verifier for RLBWT_BOUNDED_EVIDENCE_V1
- JSON schema/spec document
- mini CI fixture
- integration with existing Audit Artifact V0 / Evidence Case V1 tools
- stable server protocol documentation
- corpus/path portability cleanup

## Next target

Create replay verifier:

    tools/verify_rlbwt_bounded_evidence_v1.py

Minimum replay checks:

- runtime file hashes match artifact
- source corpus hash matches artifact
- query hash matches artifact
- server recomputes same FM interval/count/bounded offsets
- byte_check passes again
