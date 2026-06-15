# PROOF_OF_DATA_PRESENCE_POC_V1

Status:
PASS

Purpose:

Record the first GLYPH proof-of-data-presence prototype.

This prototype demonstrates two independently verifiable claims over a real public-domain corpus:

1. Membership:
   exact span is present in the committed corpus.

2. Non-membership:
   exact span is absent, bracketed by adjacent committed suffixes.

---

## Corpus

Corpus:

Pride and Prejudice public-domain text.

Use case model:

Copyright / LLM verbatim-output verification.

Question:

```text
Did these exact bytes exist in this exact corpus state?
```
---

## Artifacts

Prototype script:

research/prototypes/proof_of_data_presence/glyph_proof_v1.py

Proof artifacts:

REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_member.json
REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_absent.json

---

## Verification commands

python3 research/prototypes/proof_of_data_presence/glyph_proof_v1.py verify REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_member.json
python3 research/prototypes/proof_of_data_presence/glyph_proof_v1.py verify REFERENCE_BENCH/PROOFS/pride_poc_v1/proof_absent.json

Observed output:

VERIFY OK: MEMBERSHIP VERIFIED: span present at offset 129148, block committed
VERIFY OK: NON_MEMBERSHIP VERIFIED: span absent (bracketed by adjacent committed suffixes)

---

## Meaning

This is the first GLYPH artifact that moves beyond retrieval into proof-of-data-presence.

It is not yet production cryptography.

It is not yet a full commitment proof system.

It is an experimental prototype showing that GLYPH can emit small proof artifacts for:

* exact byte presence
* exact byte absence
* reproducible verification without re-running full retrieval

---

## Strategic interpretation

The strongest emerging application is not general search.

The strongest emerging application is:

proof of exact data presence inside a specific corpus state

This aligns with:

* evidence bundles
* commitment objects
* portable replay
* external verification
* copyright / LLM verbatim-output analysis

---

## Current status

Experimental POC:
PASS

External Verifiers:
0

Next target:

External Verifier #1
