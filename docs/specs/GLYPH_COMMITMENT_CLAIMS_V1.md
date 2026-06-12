# GLYPH_COMMITMENT_CLAIMS_V1

Status:
DRAFT

Purpose:

Define the exact statements that a GLYPH Commitment Proof can prove.

Rule:

Every claim must be:

- deterministic
- byte-exact
- independently verifiable
- machine-checkable

--------------------------------------------------

CLAIM TYPE 1

PRESENCE

Statement:

Pattern P exists in corpus C
at offset O.

Verifier learns:

- corpus commitment
- offset
- proof

Verifier can conclude:

TRUE or FALSE

--------------------------------------------------

CLAIM TYPE 2

ABSENCE

Statement:

Pattern P does not exist
in corpus C.

Verifier learns:

- corpus commitment
- proof

Verifier can conclude:

TRUE or FALSE

--------------------------------------------------

CLAIM TYPE 3

EXACT COUNT

Statement:

Pattern P occurs exactly N times
in corpus C.

Verifier learns:

- corpus commitment
- count
- proof

Verifier can conclude:

TRUE or FALSE

--------------------------------------------------

CLAIM TYPE 4

INTERVAL

Statement:

Pattern P maps to suffix interval [L,R).

Verifier learns:

- interval
- proof

Verifier can conclude:

TRUE or FALSE

--------------------------------------------------

CLAIM TYPE 5

CORPUS IDENTITY

Statement:

Corpus fingerprint F
corresponds to corpus C.

Verifier learns:

- fingerprint
- proof

Verifier can conclude:

TRUE or FALSE

--------------------------------------------------

CLAIM TYPE 6

INDEX IDENTITY

Statement:

Suffix array SA
belongs to corpus commitment C.

Verifier learns:

- commitment
- proof

Verifier can conclude:

TRUE or FALSE

--------------------------------------------------

NOT SUPPORTED

Semantic similarity

Embeddings

Meaning equivalence

Approximate matching

Fuzzy matching

Ranking

LLM interpretation

--------------------------------------------------

CORE AXIOM

GLYPH proves bytes.

GLYPH does not prove meaning.
