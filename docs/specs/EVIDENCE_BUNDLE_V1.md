# EVIDENCE_BUNDLE_V1

Status:
DRAFT

Purpose:

Define a portable verification package for GLYPH evidence.

Evidence Object proves:

query -> offset -> snippet

Evidence Bundle proves:

the evidence was produced against specific corpus and index artifacts by a specific GLYPH version.

Required files:

evidence.json
bundle_manifest.json

Required manifest fields:

bundle_version
evidence_path
evidence_sha256

corpus_path
corpus_sha256

fm_path
fm_sha256

bwt_path
bwt_sha256

sa_path
sa_sha256

created_at_utc
generated_by_tool
generated_by_commit
method
index_tag
replay_command

Required hashes:

evidence_sha256
corpus_sha256
fm_sha256
bwt_sha256
sa_sha256

Required metadata:

bundle_version
created_at_utc
generated_by_tool
generated_by_commit
method
index_tag
replay_command

Non-goal:

Evidence Bundle V1 does not embed large corpus or index artifacts.

It records their fingerprints and replay path.

Verification:

A verifier must:

1. load bundle_manifest.json
2. hash evidence.json
3. compare evidence_sha256
4. load evidence.json
5. hash corpus
6. hash FM
7. hash BWT
8. hash SA
9. compare all artifact hashes
10. replay query at offset
11. return VERIFIED or MISMATCH

Result:

Evidence Bundle V1 turns GLYPH output into a portable provenance object.

It makes a retrieval result independently replayable and fingerprint-verifiable across machines.