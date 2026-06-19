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
    "sha256": "string",
    "length_bytes": 0
  },
  "result": {
    "match_count": 0,
    "offsets": []
  },
  "verification": {
    "command": "string",
    "reproduce_status": "PASS|FAIL|UNKNOWN"
  }
}
```

## Meaning

A valid V0 artifact means:

The exact query bytes were searched against the declared corpus/index state, and the recorded result can be reproduced by running the verification command.

## Non-Goals

V0 does not claim:

- legal admissibility
- proof of truth
- zero-knowledge privacy
- authenticated FM-index correctness
- complete cryptographic non-membership
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