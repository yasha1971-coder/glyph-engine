# EVIDENCE_BUNDLE_V1

Status:
DRAFT

Purpose:

Define a portable verification package for GLYPH evidence.

Evidence Object proves:

query -> offset -> snippet

Evidence Bundle proves:

the evidence was produced against specific corpus and index artifacts.

Required files:

evidence.json

Required hashes:

corpus_sha256
fm_sha256
bwt_sha256
sa_sha256

Required metadata:

created_at_utc
glyph_commit
method
index_tag
replay_command

Non-goal:

Evidence Bundle V1 does not embed large corpus or index artifacts.

It records their fingerprints and replay path.

Verification:

A verifier must:

1. load evidence.json
2. hash corpus
3. hash FM
4. hash BWT
5. hash SA
6. compare hashes
7. replay query at offset
8. return VERIFIED or MISMATCH

Result:

Evidence Bundle V1 turns GLYPH output into a portable provenance object.
