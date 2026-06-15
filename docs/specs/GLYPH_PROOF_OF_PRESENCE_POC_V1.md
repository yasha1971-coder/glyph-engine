## GLYPH_PROOF_OF_PRESENCE_POC_V1

## Status

Research Prototype

Not Production

---

## Purpose

Document the first proof-of-data-presence prototype built on top of GLYPH retrieval primitives.

This document describes an experimental proof mechanism, its capabilities, limitations, and threat model.

It does not describe a production proof system.

---

## Scope

This prototype explores whether a committed corpus can emit small portable artifacts that allow independent verification of:

1. Exact span membership
2. Exact span non-membership

without re-running the full retrieval pipeline.

The goal is to evaluate feasibility, not to provide production-grade cryptographic guarantees.

---

## Corpus Commitment

The prototype builds a Merkle commitment over fixed corpus blocks.

The commitment binds proof artifacts to a specific corpus state.

Verification succeeds only when the proof matches the committed corpus root.

Current implementation is intended for experimentation and research.

---

## Membership Proof

Current capability

* exact span present
* verifier checks commitment consistency
* verifier checks witness consistency
* verifier verifies that the claimed span exists inside the committed witness material

Current limitation

* proof assumes the span is fully contained within a single committed block
* cross-block spans are not supported
* large spans may require future witness extensions

Security interpretation

A valid membership proof demonstrates that the claimed span existed within the committed witness material associated with the committed corpus state.

---

## Non-Membership Proof

Current capability

* exact span absent
* verifier checks witness consistency
* verifier checks commitment consistency
* verifier checks that the claimed span does not appear inside the provided witness material

Current limitation

The v1 non-membership witness is not yet cryptographically bound to a committed suffix-array ordering.

This means the verifier can validate the local witness material but cannot yet prove that the supplied neighboring suffixes are truly adjacent within the global committed suffix array.

A malicious prover could theoretically omit an intermediate suffix that would invalidate the absence claim.

Therefore this prototype must not be treated as a production-grade authenticated non-membership proof.

Correct interpretation

v1 demonstrates the shape of a portable absence witness.
v1 does not yet provide a complete authenticated
non-membership proof.

A production version would require an authenticated suffix-array structure or another commitment mechanism that binds:

* suffix ordering
* witness position
* witness adjacency
* corpus commitment

---

## Privacy Considerations

Witness artifacts contain neighboring corpus fragments.

This is acceptable for public corpora and public benchmark datasets.

It may be unsuitable for private corpora because proof artifacts can disclose surrounding corpus content.

Future versions may require privacy-preserving witness structures.

---

## Threat Model

Detects

* modification of committed corpus blocks
* modification of Merkle commitments
* modification of claimed spans
* modification of witness contents
* naive proof forgery

Does Not Yet Guarantee

* authenticated suffix-array adjacency
* formally secure non-membership proofs
* transparency-log equivalence
* zero-knowledge properties
* privacy preservation
* adversarial completeness guarantees

---

## Reproducible Example

Prototype implementation

research/prototypes/proof_of_data_presence/glyph_proof_v1.py

Proof artifacts

REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_member.json
REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_absent.json

Verification

python3 research/prototypes/proof_of_data_presence/glyph_proof_v1.py verify \
REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_member.json
python3 research/prototypes/proof_of_data_presence/glyph_proof_v1.py verify \
REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_absent.json

Expected result

VERIFY OK

---

## What This Prototype Does NOT Claim

This prototype does not claim:

* production cryptographic security
* authenticated suffix-array proofs
* transparency-log equivalence
* formally verified non-membership
* privacy-preserving proofs
* verifiable computation
* zero-knowledge proofs

---

## Future Work

Potential directions:

* authenticated suffix-array witnesses
* stronger non-membership guarantees
* witness-adjacency binding
* cross-block span support
* privacy-preserving witness structures
* corpus-scale commitment systems
* portable evidence bundles

---

## Relation to Prior Art

This prototype is conceptually related to:

* authenticated data structures
* Merkle commitment systems
* transparency logs
* verifiable storage systems
* proof-carrying evidence systems

It does not claim novelty relative to those fields.

The current goal is only to explore how GLYPH retrieval primitives can be combined with portable proof artifacts.

---

## Relation to GLYPH

GLYPH remains a deterministic byte-exact retrieval system.

This prototype explores one possible future research direction built on top of those retrieval primitives.

The proof prototype should be considered an experimental layer above retrieval, not part of the current verified retrieval core.