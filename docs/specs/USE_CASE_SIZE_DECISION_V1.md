# USE CASE SIZE DECISION V1

Status:
Decision checkpoint after LATENCY_SCALING_LAW_V1 and SA32_BOUNDARY_AUDIT_V1.

## Context

GLYPH currently has a verified persistent FM latency law through 2GB corpus scale:

512MB:
LAW_CONFIRMED

1GB:
LAW_CONFIRMED

2GB:
LAW_CONFIRMED

4GB is not yet a latency frontier.

4GB is currently a representation frontier because the canonical build_sa path is int32-based.

## Key Distinction

Latency ceiling:
runtime / retrieval physics

Representation ceiling:
suffix-array integer width / corpus addressability

These must not be mixed.

## Current Verified Law

For resident FMBINv2 indexes with checkpoint_step=256,
GLYPH preserves microsecond-scale steady-state exact substring retrieval through at least 2GB corpus scale.

## Strategic Question

Before choosing SA64 or segmentation, answer:

What corpus size does the target user actually need?

Especially for:
- log forensics
- debugging analysis
- exact byte retrieval
- RAG exact prefiltering
- binary/text corpus inspection

## Option A: Canonical <=2GB GLYPH

Meaning:
Accept <=2GB shard/corpus as the canonical verified unit.

Advantages:
- already verified
- microsecond-scale latency law confirmed
- avoids SA64 complexity
- avoids cross-shard boundary semantics
- strong for log forensics if real workloads fit <=2GB shards

Risk:
- not a monolithic >2GB engine
- must explain boundary clearly

Best if:
target use case naturally operates on bounded shards/log windows.

## Option B: Segmented continuation

Meaning:
Keep each shard under safe SA32/int32 boundary.

Advantages:
- avoids monolithic SA64
- supports larger total datasets
- aligns with sharded log/archive workflows

Risks:
- cross-shard boundary matches require explicit semantics
- fan-out latency must be remeasured
- result merging becomes part of retrieval path

Best if:
target use case is multi-file/multi-log/multi-shard retrieval.

## Option C: SA64 monolith

Meaning:
Implement true 64-bit suffix-array path.

Advantages:
- removes 4GB ceiling honestly
- preserves monolithic semantics
- continues latency law without shard boundary issues

Risks:
- RAM cost increases
- builder complexity increases
- index format compatibility must be extended
- may solve a problem the first real users do not need

Best if:
target users require single monolithic corpora >4GB.

## Decision Rule

Do not choose based on engineering curiosity.

Choose based on:
- real corpus size distribution
- target workload
- acceptable shard semantics
- memory budget
- reproducibility
- product positioning

## Current Recommendation

Do not start SA64 migration yet.

Do not start segmented fan-out yet.

First define the target corpus-size envelope for GLYPH’s first credible use case.

Possible default:

GLYPH v0.x:
verified exact retrieval substrate for <=2GB corpus/shard units.

Future:
SA64 or segmentation chosen only after use-case evidence.
