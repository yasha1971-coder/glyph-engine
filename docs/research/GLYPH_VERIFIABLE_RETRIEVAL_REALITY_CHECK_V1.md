# GLYPH_VERIFIABLE_RETRIEVAL_REALITY_CHECK_V1

Status: Research Note / Positioning Guardrail  
Date: 2026-06-19

## Purpose

This note defines the current research position of GLYPH after multiple external/AI reality checks around verifiable retrieval, authenticated search, ZK query systems, RAG provenance, transparency logs, and data provenance systems.

Its purpose is to prevent false positioning.

This note does not claim novelty.

This note does not claim legal admissibility.

This note does not claim production cryptographic security.

It defines the current narrow engineering question:

Can GLYPH become a cheap proof-adjacent exact retrieval layer over committed corpora?

## Current GLYPH V0 State

GLYPH currently has:

- deterministic exact byte-level retrieval over static corpora
- canonical sentinel-safe FM-index pipeline
- manifest verification
- mini reproducible example
- GLYPH_AUDIT_ARTIFACT_V0 specification
- audit artifact generator
- audit artifact verifier

Current reproducible V0 path:

1. Build mini index.
2. Query exact pattern.
3. Generate audit artifact.
4. Verify audit artifact.

Current mini command path:

    ./examples/mini/run_mini.sh

    python3 tools/glyph_make_audit_artifact_v0.py \
      --index-dir examples/mini/out \
      --query error \
      --output examples/mini/out/audit_artifact_v0.json

    python3 tools/glyph_verify_audit_artifact_v0.py \
      examples/mini/out/audit_artifact_v0.json

Expected result:

    VERIFY AUDIT ARTIFACT OK

## What V0 Proves

V0 proves only a minimal reproducibility claim:

- the declared corpus file has the expected SHA-256 hash
- the declared manifest has the expected SHA-256 hash
- the declared query bytes hash to the recorded query hash
- the recorded verification command can be replayed
- the replayed FM query returns the recorded FM interval and match count

## What V0 Does Not Prove

V0 does not prove:

- legal admissibility
- truth of content
- zero-knowledge privacy
- authenticated FM-index correctness
- complete cryptographic membership
- complete cryptographic non-membership
- complete cryptographic result completeness
- adversarial soundness against a malicious prover
- production-grade proof security

V0 is a reproducible audit record, not a final proof system.

## Main Positioning Correction

GLYPH must not claim:

- nobody works on this
- no related systems exist
- GLYPH is a ZK proof system
- GLYPH is court-ready evidence
- GLYPH is a unique mathematical invention
- GLYPH already solves non-membership or completeness

The correct position is narrower:

GLYPH explores verifiable exact retrieval over committed corpora.

More precisely:

GLYPH explores whether a prebuilt exact retrieval index can produce cheap, reproducible, portable audit artifacts binding corpus state, index state, query bytes, and exact retrieval results.

## Adjacent Prior Art and Neighbor Systems

The following fields and systems must be acknowledged:

- VPM / authenticated pattern matching
- Practical Authenticated Pattern Matching
- authenticated suffix trees / authenticated suffix arrays
- IntegriDB
- vSQL
- Space and Time Proof of SQL
- Lagrange ZK MapReduce / ZK SQL
- V3DB / verifiable vector search
- VeriRAG / verifiable RAG research
- Geppetto / verifiable MapReduce demos
- Certificate Transparency
- Rekor / Trillian
- OpenTimestamps
- IPFS / Arweave
- Sigstore / SLSA / in-toto
- C2PA
- RAGShield / ProveRAG / RAG provenance systems
- substring-searchable symmetric encryption

These systems solve important adjacent problems.

They do not automatically solve GLYPH's exact target.

## System Class Comparison

### Transparency Logs

Examples:

- Certificate Transparency
- Rekor
- Trillian

They prove:

- inclusion of a known log entry
- append-only consistency
- tamper-evident logging

They do not natively prove:

- arbitrary substring search
- complete match set retrieval
- absence of a substring inside all logged data
- byte-level search over a committed corpus

Conclusion:

Transparency logs are important infrastructure analogs, not direct replacements.

### Timestamping and Content Addressing

Examples:

- OpenTimestamps
- IPFS
- Arweave

They prove:

- file/content identity
- existence at time
- immutable addressing or storage

They do not natively prove:

- exact substring retrieval
- complete search result correctness
- query execution over corpus content

Conclusion:

They can commit or preserve data, but they do not search inside it with proof artifacts.

### Verifiable SQL / Verifiable Databases

Examples:

- vSQL
- IntegriDB
- Space and Time Proof of SQL
- Lagrange ZK SQL / ZK MapReduce

They prove:

- supported SQL query correctness
- some structured query completeness
- some absence results for supported predicates

Unresolved for GLYPH target:

- arbitrary byte-level substring queries over raw corpora
- practical suffix/FM-index-style completeness artifacts
- cheap exact retrieval proof over large unstructured text
- simple external artifact understandable outside the service

Conclusion:

This is one of the strongest neighboring fields. GLYPH must compare against it carefully.

### Verifiable Vector Retrieval

Examples:

- V3DB
- VeriRAG-like systems

They prove:

- vector top-k retrieval under fixed semantics
- snapshot-bound vector results
- sometimes zero-knowledge vector search correctness

They do not prove:

- exact byte substring retrieval
- raw text occurrence completeness
- absence of byte pattern in corpus

Conclusion:

Relevant for semantic RAG, not direct replacement for exact byte-level retrieval.

### RAG Provenance and RAG Security

Examples:

- RAGShield
- ProveRAG
- S-RAG variants
- LangSmith-like tracing
- C2PA-style attestations

They provide:

- provenance
- citation trails
- document attestation
- poisoning defense
- source traceability

They do not provide:

- cryptographic exact byte search
- arbitrary substring absence proof
- portable proof that all matches were returned

Conclusion:

Relevant user pain, but not the same primitive.

### Authenticated Pattern Matching

Examples:

- VPM
- Practical Authenticated Pattern Matching
- authenticated suffix structures

This is the closest scientific prior art.

It can address:

- authenticated pattern matching
- match / mismatch verification
- proof-carrying text search

Open questions for GLYPH:

- practical implementation status
- proof generation cost
- integration with modern corpus artifacts
- relation to FM-index/BWT pipelines
- portable artifact format
- usability for external verification

Conclusion:

This must be treated as direct prior art, not ignored.

## Core Technical Gap

The target missing layer is:

committed corpus  
+ exact byte-level / substring retrieval  
+ reproducible retrieval  
+ portable audit artifact  
+ path to membership, non-membership, and completeness

The hard part is not merely proving a positive match.

The hard part is proving cheaply that:

- all matches were returned
- no match exists when result is empty
- the result is tied to a fixed corpus state
- the verifier does not need to rerun the full search
- the provider does not need to run a prohibitively expensive proof for every query

## Central Risk

If proof generation is expensive, GLYPH only moves the burden from verifier to provider.

Therefore the core question is not:

Can GLYPH produce a proof?

The core question is:

Can GLYPH produce useful reproducible exact retrieval artifacts cheaply enough over prebuilt indexes to matter in practice?

## Current Working Hypothesis

GLYPH should not be positioned as a general ZK system.

GLYPH should be positioned as:

cheap proof-adjacent exact retrieval over committed corpora.

Near-term value comes from:

- deterministic exact retrieval
- corpus hash binding
- index manifest hash binding
- query byte hash binding
- replayable verification command
- portable artifact
- external reproducibility

Long-term value may come from:

- membership proofs
- non-membership proofs
- completeness proofs
- authenticated suffix/FM-index structures
- optional integration with external timestamping or transparency systems

## Claims GLYPH Can Safely Make Today

GLYPH can currently claim:

- GLYPH has a minimal reproducible audit artifact V0.
- V0 binds corpus hash, manifest hash, query hash, FM interval, and match count.
- V0 can be independently replay-verified in the mini example.
- V0 is a research artifact toward verifiable exact retrieval over committed corpora.

GLYPH should not yet claim:

- cryptographic proof of absence
- cryptographic proof of completeness
- court-ready evidence
- zero-knowledge retrieval
- production adversarial security
- superiority over VPM / IntegriDB / Proof of SQL / Lagrange

## Next Engineering Step

The immediate engineering step is complete for V0:

- artifact specification
- artifact generator
- artifact verifier
- mini documentation

The next engineering step should not be heavy ZK.

The next engineering step should be one of:

1. Add offsets to the audit artifact by integrating locate output.
2. Add a regression test for audit artifact generation and verification.
3. Add a research comparison note against VPM / IntegriDB / Proof of SQL / Lagrange / V3DB / CT.
4. Add a public README section that positions GLYPH carefully.

## Next Research Step

The next research step is to manually verify the strongest prior-art neighbors:

1. Practical Authenticated Pattern Matching.
2. VPM.
3. Secure Hashing-Based Verifiable Pattern Matching.
4. IntegriDB.
5. vSQL.
6. Space and Time Proof of SQL.
7. Lagrange ZK MapReduce.
8. Geppetto substring / MapReduce demo.
9. V3DB / VeriRAG.
10. Certificate Transparency / Rekor.

For each system, answer:

- what is committed?
- what query class is supported?
- what does the proof actually prove?
- does it support raw byte substring queries?
- does it prove absence?
- does it prove completeness?
- what is prover cost?
- is the artifact portable?
- is it implemented or only academic?

## Current Decision

Continue GLYPH, but with stricter language.

Do not sell GLYPH as "the first" or "the only".

Build the smallest real artifacts and let the comparison table speak.

Current axis:

verifiable exact retrieval over committed corpora.

