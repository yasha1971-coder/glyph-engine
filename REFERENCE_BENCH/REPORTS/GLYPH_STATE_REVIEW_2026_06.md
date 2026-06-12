GLYPH_STATE_REVIEW_2026_06

Status:
REVIEW

Date:
2026-06

⸻

CURRENT VALIDATED CAPABILITIES

1. Sentinel-safe FM retrieval validated.
2. Exact locate via suffix array validated.
3. Retrieval V1 validated.
4. Evidence Object V1 implemented.
5. Evidence Export V1 implemented.
6. Evidence Replay V1 implemented.
7. Corpus Fingerprint V1 implemented.
8. Evidence Bundle V1 implemented.
9. Bundle Replay V1 implemented.
10. Portable Bundle Replay V1 validated outside repository.
11. Independent artifact portability demonstrated.

⸻

CURRENT RESEARCH TRACKS

Track A:
Retrieval Provenance Engine

Status:
Validated

Track B:
Corpus Commitments

Status:
Specification Phase

Track C:
Public Audit Layer

Status:
Research

Track D:
Corpus Genealogy

Status:
Research

Track E:
Multi-Auditor Trust

Status:
Research

⸻

WHAT IS ACTUALLY NOVEL

Not FM-index.

Not suffix arrays.

Not byte-exact retrieval.

Not Merkle commitments.

Not authenticated pattern matching.

Potentially novel:

1. Portable provenance chain.

Evidence
→ Bundle
→ Replay
→ Portable Replay

2. Public-auditable retrieval provenance.
3. Commitment + Audit + Provenance integration.
4. Corpus genealogy over deterministic retrieval artifacts.
5. Engine-independent provenance verification workflow.
6. Practical reference implementation for verifiable retrieval.

⸻

WHAT IS PRIOR ART

Known before GLYPH:

FM-index

Suffix arrays

Merkle trees

Certificate Transparency

Authenticated dictionaries

Authenticated pattern matching
(Papadopoulos et al., 2015)

Verifiable databases

Cryptographic commitments

SNARK/STARK verification systems

⸻

BIGGEST RISKS

1. Reinventing known literature.
2. Claiming novelty where prior art exists.
3. Producing specifications without executable validation.
4. Over-expanding into generic RAG.
5. Losing focus on provenance.
6. Building complexity before proving demand.

⸻

NEXT EXPERIMENTS

1. Commitment Builder V1

Output:

* Corpus Root
* SA Root
* Commitment Root

2. Commitment Object V1
3. Commitment Replay Validation
4. Merkle Presence Proof Prototype
5. Merkle Absence Proof Prototype
6. Audit Attestation Prototype

⸻

STOP DOING

Building generic retrieval features.

Discussing embeddings.

Discussing ANN search.

Discussing chatbot products.

Adding features without validation.

⸻

START DOING

Produce executable artifacts.

Measure proof sizes.

Measure verifier cost.

Measure commitment build cost.

Validate portability.

Validate audit workflow.

⸻

WORKING THESIS

GLYPH is evolving from:

Deterministic Retrieval Engine

toward

Verifiable Retrieval Infrastructure.

⸻

EPOCH MARKER

PORTABLE PROVENANCE VERIFIED

COMMITMENT RESEARCH ACTIVE