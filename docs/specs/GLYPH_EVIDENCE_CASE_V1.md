GLYPH_EVIDENCE_CASE_V1

Status: Draft Specification
Date: 2026-06-19

Purpose

GLYPH_EVIDENCE_CASE_V1 is a human-readable layer built on top of GLYPH_AUDIT_ARTIFACT_V0.

Audit Artifact V0 is optimized for reproducibility.

Evidence Case V1 is optimized for inspection.

The goal is to allow a reviewer to understand:

* what corpus was searched
* what query was executed
* where matches occurred
* what surrounding context existed

without manually replaying the query.

Relationship to Audit Artifact

Audit Artifact V0:

* corpus hash
* manifest hash
* query hash
* FM interval
* offsets
* replay command

Evidence Case V1:

* references an Audit Artifact
* preserves corpus identity
* preserves query identity
* adds human-readable snippets
* adds per-offset evidence records

Evidence Case V1 does not replace Audit Artifact V0.

It is derived from it.

Minimum Fields

{
  "case_version": "GLYPH_EVIDENCE_CASE_V1",
  "source_artifact": "path",
  "corpus": {},
  "query": {},
  "result_summary": {},
  "evidence_records": []
}

Evidence Record

Each evidence record should contain:

* offset
* matched bytes
* matched text (when decodable)
* snippet boundaries
* snippet text
* byte_check result

Example:

{
  "offset": 37,
  "match_text": "error",
  "byte_check": true
}

Meaning

A valid Evidence Case V1 means:

The evidence records were generated from a reproducible Audit Artifact and the recorded offsets correspond to matching corpus bytes.

Non-Goals

Evidence Case V1 does not claim:

* legal admissibility
* cryptographic completeness
* non-membership proofs
* authenticated search proofs
* zero-knowledge security

Positioning

Audit Artifact V0 answers:

“Can this result be reproduced?”

Evidence Case V1 answers:

“What did the result actually look like?”
