POST_SA32_EVIDENCE_REQUIREMENTS_V1

Status: OPEN

Date: 2026-05-31

⸻

Purpose

No post-SA32 architecture decision shall be made without supporting evidence.

Current candidates:

* Canonical 2GB Shards
* SA64 Monolith
* Hybrid Architecture

The goal is to gather evidence before implementation.

⸻

Required Evidence

E1. Real Corpus Size Distribution

Question:

What corpus sizes do actual users need?

Data to collect:

* log archives
* forensic datasets
* code corpora
* document corpora
* research datasets

Need:

* median corpus size
* p95 corpus size
* maximum observed corpus size

Decision impact:

If most real deployments are ≤2GB, SA64 may be unnecessary.

⸻

E2. Query Locality

Question:

How often do queries cross shard boundaries?

Need:

* single-shard query frequency
* multi-shard query frequency

Decision impact:

If most queries are local, sharding becomes cheaper.

⸻

E3. Startup Constraints

Question:

What startup time is acceptable?

Need:

* acceptable cold-start
* acceptable warm restart

Decision impact:

Large monoliths increase startup cost.

⸻

E4. Memory Budget

Question:

How much RAM is realistically available?

Need:

* workstation deployments
* server deployments
* cloud deployments

Decision impact:

SA64 doubles suffix-array footprint.

⸻

E5. Operational Complexity

Question:

What is easier to maintain?

Need:

* shard management cost
* monolith management cost
* backup cost
* recovery cost

Decision impact:

Operational burden may dominate theoretical performance.

⸻

E6. Cross-Shard Semantics

Question:

Are cross-shard matches required?

Need:

* frequency of boundary-crossing matches
* acceptable handling strategy

Decision impact:

Determines viability of canonical sharding.

⸻

Current State

Evidence gathered:

* latency scaling validated
* latency regime validated
* SA32 boundary audited

Evidence missing:

* real-world corpus requirements
* operational requirements

⸻

Rule

Architecture follows use case.

Use case does not follow architecture.

No implementation work should begin until evidence is collected.
