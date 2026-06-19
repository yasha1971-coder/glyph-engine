# GLYPH_AUDIT_ARTIFACT_V0
Status: Draft / Research Prototype
## Purpose
GLYPH_AUDIT_ARTIFACT_V0 defines the first portable audit artifact for reproducible exact retrieval over a fixed committed corpus.
This is not a legal proof.
This is not a zero-knowledge proof.
This is not a production authenticated search protocol.
It is a minimal reproducible record showing:
- which corpus was searched
- which index/manifest was used
- which exact query bytes were searched
- which offsets were returned
- which command can reproduce the result
## Positioning
GLYPH is positioned as:
verifiable exact retrieval over committed corpora
Not:
- fast grep
- generic search engine
- vector provenance tool
- blockchain storage layer
- court-ready evidence system
## Artifact Goals
The artifact should allow an independent reviewer to answer:
1. What corpus state was searched?
2. What GLYPH index/manifest was used?
3. What exact query was executed?
4. What offsets were returned?
5. Can the result be reproduced?
## Minimum Fields
```json
{
  "artifact_version": "GLYPH_AUDIT_ARTIFACT_V0",
  "created_at_utc": "string",
  "glyph_version": "string",
  "corpus": {
    "path": "string",
    "sha256": "string",
    "size_bytes": 0
  },
  "index_manifest": {
    "path": "string",
    "sha256": "string"
  },
  "query": {
    "encoding": "hex",
    "hex": "string",
    "sha256": "string",
    "length_bytes": 0
  },
  "result": {
    "match_count": 0,
    "fm_interval": [0, 0],
    "offsets": [],
    "offset_mode": "locate_backend_v2|not_available"
  },
  "verification": {
    "command": "string",
    "reproduce_status": "PASS|FAIL|UNKNOWN",
    "returncode": 0
  }
}
```

## Meaning

A valid V0 artifact means:

The exact query bytes were searched against the declared corpus/index state, and the recorded result can be reproduced by running the verification command.

In V0, when offsets are present, the verifier also checks that each recorded offset points to corpus bytes exactly equal to the query bytes.

For example, if the query is `error` and the artifact records offsets `[0, 37]`, verification checks:

- `corpus[0:5] == b"error"`
- `corpus[37:42] == b"error"`

This is still a reproducibility and byte-check claim, not a cryptographic completeness proof.

## Non-Goals

V0 does not claim:

- legal admissibility
- proof of truth
- zero-knowledge privacy
- authenticated FM-index correctness
- complete cryptographic non-membership
- cryptographic completeness of all offsets
- production-grade proof system

## Future Directions

Possible later versions may add:

- corpus commitment roots
- Merkle inclusion paths
- index commitment
- membership proofs
- non-membership proofs
- completeness proofs
- independent verifier CLI
- timestamp anchoring
- integration with immutable storage systems

## Current Rule

Build the smallest reproducible audit artifact first.

Do not build large cryptographic infrastructure until V0 is working and externally understandable.