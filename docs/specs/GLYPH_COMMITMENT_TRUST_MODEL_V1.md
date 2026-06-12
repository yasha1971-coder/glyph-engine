# GLYPH_COMMITMENT_TRUST_MODEL_V1

Status:
DRAFT

Purpose:

Define exactly who must be trusted,
who must not be trusted,
and what can be verified independently.

--------------------------------------------------

PARTIES

Publisher

Auditor

Verifier

--------------------------------------------------

PUBLISHER

Role:

Creates:

- corpus
- suffix array
- commitment
- proofs

Publisher is NOT automatically trusted.

Publisher may be:

- honest
- mistaken
- malicious

--------------------------------------------------

AUDITOR

Role:

Rebuilds deterministic artifacts.

Checks:

- corpus fingerprint
- suffix array correctness
- commitment correctness

Auditor produces:

AUDIT ATTESTATION

Auditor is independent from Publisher.

--------------------------------------------------

VERIFIER

Role:

Checks proof.

Verifier should not need:

- original GLYPH engine
- original machine
- original repository

Verifier consumes:

- commitment
- proof
- audit record

Verifier outputs:

TRUE
or
FALSE

--------------------------------------------------

TRUST ASSUMPTIONS

Trusted:

SHA256

Deterministic build specification

Public commitment format

Untrusted:

Publisher

Network

Storage provider

Transport layer

Repository host

--------------------------------------------------

CURRENT MODEL

Evidence Replay

Trust path:

Publisher
→ Evidence
→ Replay
→ Verification

--------------------------------------------------

FUTURE MODEL

Commitment Verification

Trust path:

Publisher
→ Commitment

Auditor
→ Audit Attestation

Verifier
→ Independent Verification

--------------------------------------------------

MALICIOUS PUBLISHER CASE

Publisher may:

- modify corpus
- modify index
- fabricate absence claims
- fabricate count claims

Goal:

Detection must occur.

--------------------------------------------------

MALICIOUS AUDITOR CASE

Single auditor may fail.

System should allow:

Multiple independent auditors.

--------------------------------------------------

MALICIOUS VERIFIER CASE

Verifier cannot create truth.

Verifier only evaluates proof.

--------------------------------------------------

CORE PRINCIPLE

Trust should move away from software.

Trust should move toward proofs.

--------------------------------------------------

LONG TERM GOAL

A verifier should eventually validate:

Presence

Absence

Count

Interval

without running GLYPH retrieval.

--------------------------------------------------

NON-GOALS

Trusting authority.

Trusting vendor.

Trusting company.

Trusting repository.

Trusting AI.

--------------------------------------------------

AXIOM

Do not trust the producer.

Verify the claim.
