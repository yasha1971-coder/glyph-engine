GLYPH_PROOF_OF_PRESENCE_POC_V1

Status:
Research Prototype

Not Production

Purpose

Document the first proof-of-data-presence prototype built on top of GLYPH retrieval primitives.

This document describes an experimental proof mechanism and its limitations.

It does not describe a production proof system.

⸻

Scope

This prototype explores whether a committed corpus can emit small artifacts that allow independent verification of:

1. Exact span membership
2. Exact span non-membership

without re-running the full retrieval pipeline.

⸻

Corpus Commitment

The prototype builds a Merkle commitment over fixed corpus blocks.

The commitment binds proof artifacts to a specific corpus state.

Current implementation is intended for experimentation only.

⸻

Membership Proof

Current capability:

* exact span present
* verifier checks commitment consistency
* verifier checks witness consistency

Current limitation:

* proof assumes the span is contained within a single committed block
* cross-block spans are not supported

⸻

Non-Membership Proof

Current capability:

* exact span absent
* verifier checks witness consistency

Current limitation:

The witness is not yet cryptographically bound to a committed suffix-array ordering.

A malicious prover could theoretically hide intermediate suffixes.

Therefore this prototype must not be treated as a production-grade non-membership proof.

⸻

Privacy Considerations

Witness artifacts contain neighboring corpus fragments.

This is acceptable for public corpora.

It may be unsuitable for private corpora.

⸻

Reproducible Example

Prototype implementation:

research/prototypes/proof_of_data_presence/glyph_proof_v1.py

Artifacts:

REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_member.json

REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_absent.json

Verification:

python3 research/prototypes/proof_of_data_presence/glyph_proof_v1.py verify REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_member.json

python3 research/prototypes/proof_of_data_presence/glyph_proof_v1.py verify REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_absent.json

Expected result:

VERIFY OK

⸻

What This Prototype Does NOT Claim

This prototype does not claim:

* production cryptographic security
* authenticated suffix-array proofs
* transparency-log equivalence
* formally verified non-membership
* privacy-preserving proofs

⸻

Future Work

Potential directions:

* authenticated suffix-array witnesses
* stronger non-membership guarantees
* cross-block span support
* privacy-preserving witness structures
* corpus-scale commitment systems

⸻

Relation to GLYPH

GLYPH remains a deterministic byte-exact retrieval system.

This prototype explores one possible future research direction built on top of those retrieval primitives.
