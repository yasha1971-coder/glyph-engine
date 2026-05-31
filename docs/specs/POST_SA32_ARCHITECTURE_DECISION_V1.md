POST_SA32_ARCHITECTURE_DECISION_V1

Status: OPEN

Date: 2026-05-31

⸻

Purpose

Retrieval Physics Milestone is complete.

The next question is no longer:

* correctness
* latency
* repeatability

Those have been validated.

The remaining question is:

How should GLYPH scale beyond the current SA32 boundary?

⸻

Current State

Validated:

* FM correctness
* sentinel-safe invariant
* persistent query architecture
* steady-state latency
* repeatability
* latency regime
* scaling law (512MB → 2GB)

Observed result:

Steady-state retrieval latency remains effectively constant across tested corpus sizes.

Current boundary:

SA32 representation width.

Not retrieval latency.

⸻

Decision Criteria

Future architecture should preserve:

1. Deterministic retrieval
2. Byte-exact semantics
3. Manifest-first discipline
4. Reproducibility
5. Operational simplicity
6. Low steady-state latency

The objective is not maximum corpus size.

The objective is useful corpus size with minimal complexity.

⸻

Option A

Canonical 2GB Shards

Architecture:

Corpus
  ↓
2GB shard
2GB shard
2GB shard

Properties:

Pros:

* preserves current validated architecture
* preserves latency regime
* simple operational model
* independent rebuilds
* independent verification
* limited blast radius

Cons:

* routing required
* fan-out required
* shard boundary handling required
* global retrieval becomes distributed

Assessment:

Lowest engineering risk.

⸻

Option B

SA64 Monolith

Architecture:

Large Corpus
      ↓
SA64
      ↓
BWT
      ↓
FM

Properties:

Pros:

* single global index
* no routing layer
* no shard management
* conceptually simple

Cons:

* larger memory footprint
* larger build cost
* larger operational blast radius
* unknown latency effects
* significant implementation work

Assessment:

Highest engineering cost.

⸻

Option C

Hybrid Architecture

Architecture:

Global Corpus
      ↓
Shard Registry
      ↓
SA32 Shards
      ↓
Local FM Retrieval

Each shard remains independently valid.

Global layer performs:

* routing
* aggregation
* provenance

Local layer performs:

* retrieval

Properties:

Pros:

* preserves current latency regime
* preserves validated SA32 path
* scalable beyond single corpus boundary
* smaller operational units

Cons:

* additional coordination layer
* more metadata

Assessment:

Potential long-term direction.

⸻

Use Case Question

Before selecting an architecture:

Determine actual corpus requirements.

Questions:

1. Do target users require >2GB monolithic corpora?
2. Are real-world datasets naturally sharded already?
3. Is operational simplicity more valuable than theoretical scale?
4. Does the use case benefit from independent shard lifecycle management?

Architecture should follow use case requirements.

Not theoretical limits.

⸻

Current Recommendation

No implementation work yet.

First establish:

* real corpus size distributions
* operational requirements
* expected deployment model

Then choose:

* Canonical 2GB Shards
* SA64 Monolith
* Hybrid Architecture

⸻

Current Verdict

Retrieval Physics Milestone:

COMPLETED

Post-SA32 Architecture:

OPEN

No architecture has been selected yet.
