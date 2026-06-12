# GLYPH_COMMITMENT_PHASE_0

Status:
DRAFT

Purpose:

Define the formal semantics and trust model for the future GLYPH Corpus Commitment Layer.

Core correction:

The goal is not to invent verifiable pattern matching from scratch.

Prior art already exists.

The GLYPH goal is:

hash-only, portable, replay-compatible, audit-amortized verification for byte-exact retrieval over static corpora.

Current chain:

Retrieval
→ Evidence
→ Bundle
→ Portable Replay

Next chain:

Evidence
→ Bundle
→ Commitment
→ Audit
→ Case

Core principle:

Replay is not removed.

Replay is amortized.

Old model:

every verifier replays every claim.

New model:

auditors replay each corpus once.

verifiers validate proofs forever.

Claim Types:

1. Presence Claim

This exact byte sequence occurs in this exact corpus at this exact offset.

2. Absence Claim

This exact byte sequence does not occur in this exact corpus.

3. Count Claim

This exact byte sequence occurs exactly N times in this exact corpus.

4. Interval Claim

This query corresponds to suffix-array interval [l, r).

Trust Model:

Publisher:
creates corpus, index, commitment.

Auditor:
rebuilds deterministic index from corpus and verifies commitment correctness.

Verifier:
checks proof against commitment without rerunning full GLYPH retrieval.

Threats:

1. Malformed suffix array.
2. Wrong corpus fingerprint.
3. Incorrect absence proof.
4. Incorrect count proof.
5. Encoding mismatch.
6. Mutable corpus state.
7. Overclaiming legal or semantic meaning.

Non-goals:

No semantic search.
No embeddings.
No ANN.
No ZK requirement for V1.
No confidential corpus proof.
No blockchain-first design.
No patent-dependent accumulator design.

Preferred V1 Direction:

Hash-only Merkle corpus + Merkle suffix array.

Reason:

- compatible with GLYPH
- implementable by one founder
- no trusted setup
- no pairings
- no patent exposure
- independently verifiable
- portable across machines

Next Required Design Documents:

1. GLYPH_COMMITMENT_CLAIMS_V1.md
2. GLYPH_COMMITMENT_TRUST_MODEL_V1.md
3. GLYPH_COMMITMENT_ARCHITECTURE_V1.md
4. GLYPH_COMMITMENT_AUDIT_PROTOCOL_V1.md
