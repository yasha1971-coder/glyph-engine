# GLYPH Engine Overview

Status:

    CURRENT VERIFIED ARCHITECTURE OVERVIEW

This document is a non-normative map of the current GLYPH architecture.

Normative behavior is defined by the specifications under:

    docs/specs/

Executable truth is established by:

    ./verify.sh

## Purpose

GLYPH is a verifiable exact-byte retrieval system for committed corpora.

Its central property is not relevance ranking.

Its central property is that an exact retrieval claim can be reproduced
against an identified corpus using deterministic semantics and independently
checked evidence.

GLYPH is not currently:

- a relevance-ranked search engine;
- a vector database;
- a fuzzy matcher;
- a regular-expression engine;
- a mutable transactional database;
- a distributed storage platform;
- a legal proof system.

## Current verified architecture

The required verification chain has four intentionally separate layers:

    reference semantics
    -> compiled binary-safe runtime
    -> operator workflow and replay
    -> Embedded C ABI pre-freeze contract

The current required closure is:

    GLYPH PROOF GRAPH OK
    GLYPH RUNTIME CONFORMANCE OK
    GLYPH OPERATOR CONFORMANCE OK
    GLYPH EMBEDDED I0 CONTRACT VERIFY PASS
    VERIFY OK

Each layer has a different responsibility.

## Reference semantics — P1 through P12

The reference layer defines the exact retrieval laws independently of an
optimized runtime implementation.

It covers properties including:

- suffix and suffix-array validity;
- suffix/BWT relation;
- exact FM backward search;
- exact locate coordinates;
- document-boundary behavior;
- deterministic evidence obligations;
- mutation rejection.

The reference layer acts as the semantic oracle for compiled implementations.

Future optimizations must preserve these semantics.

## Compiled runtime — R0 through R6

The compiled runtime conformance graph separates the historical sentinel-safe
baseline from the current binary-safe path.

### R0 — historical sentinel-safe baseline

R0 preserves and checks the earlier physical-sentinel path:

- source `0x00` is rejected;
- physical `0x00` is appended as the sentinel.

This path remains useful as historical and differential evidence.

It is not the complete byte-domain architecture of the current verified
binary-safe runtime.

### R1 and R2 — binary-safe count and locate

The current binary-safe C++ runtime supports source bytes:

    0x00 through 0xFF

Its internal sentinel model uses:

    virtual sentinel 256

The FM alphabet therefore contains:

    257 symbols

The sentinel is not a source byte and cannot collide with corpus content.

The binary-safe runtime provides:

- exact count;
- exact locate;
- deterministic failure behavior;
- checked runtime artifacts;
- parity with the reference semantics.

### R3 — multi-document runtime

A verified multi-document corpus is an ordered collection of independent
documents.

Each logical document is indexed independently.

Canonical result coordinates are:

    (document_id, document_offset)

Ordering is:

    document_id ascending
    then document_offset ascending

Document identity is positional within the committed ordered corpus.

Empty documents and duplicate documents remain represented explicitly.

A match cannot cross:

- a document boundary;
- an internal virtual sentinel;
- two independently indexed documents.

Physical concatenation is not the matching oracle for a multi-document
corpus.

### R4 through R6 — evidence, bundles and replay

The runtime evidence path binds exact results to identified runtime inputs.

The verified path includes:

- runtime evidence artifacts;
- source-byte checks;
- deterministic coordinate records;
- replay through the compiled C++ runtime;
- self-contained evidence bundles;
- copied replay outside the source repository.

Replay must recompute relevant facts.

It must not trust a stored success flag such as `byte_check=true` without
checking the committed bytes again.

## Operator workflow — O1 through O6

The operator layer transforms an ordinary filesystem corpus into a
reproducible exact-retrieval workflow.

Conceptually:

    filesystem corpus
    -> canonical source manifest
    -> binary-safe runtime artifacts
    -> exact query
    -> canonical result
    -> evidence bundle
    -> independent replay

The operator layer defines:

- deterministic filesystem enumeration;
- committed source-manifest identity;
- path and file metadata rules;
- binary-safe query transport;
- exact count and bounded locate;
- canonical multi-document aggregation;
- self-contained evidence bundles;
- fail-closed replay.

The operator may add display metadata such as source paths.

Display metadata does not replace canonical coordinates.

## Corpus and document identity

The operator source manifest binds the committed ordered document collection.

For each document it records identity material including:

- canonical relative path bytes;
- byte length;
- source hash;
- canonical manifest position.

The canonical document identifier is derived from the committed document
order.

Two byte-identical documents at different manifest positions remain distinct
documents.

Reordering documents changes the committed corpus identity.

## Query and bounded locate semantics

Exact match count is complete for the committed corpus.

Bounded locate limits the number of returned coordinates, not the correctness
of the total count.

For a limit `max_offsets`, the returned coordinates are the canonical prefix
of the complete ordered coordinate sequence.

A successful bounded result must distinguish:

- complete match count;
- returned coordinate count;
- whether the coordinate list is bounded;
- whether all offsets were returned.

## Failure philosophy

GLYPH prefers explicit failure to silent incompleteness.

A successful result must not silently omit required corpus units, documents,
runtime artifacts or evidence inputs.

Malformed, incompatible, missing or mutated artifacts must produce defined
failure rather than an apparently valid smaller result.

## Embedded API V1 — I0 status

The public C header and normative Embedded I0 contract documents exist.

They define a pre-freeze integration contract covering:

- fixed-width C ABI types;
- V1 public layouts;
- ownership and lifetime;
- caller-owned bounded locate output;
- failure-output semantics;
- document identity;
- concurrency and close barriers;
- resource and timeout boundaries;
- mmap trust assumptions;
- hostile-file parsing;
- signed-statement policy.

Current status:

    DRAFT_NOT_FROZEN

The shared-library implementation has not started.

The published second-review package is an immutable review snapshot.

Publishing that review package does not freeze the ABI.

No stable Embedded E1 implementation may be inferred from the existence of
the I0 contract.

## Legacy segmented architecture

GLYPH v0.x and v0.2 previously experimented with:

- physical appended `0x00` sentinels;
- source corpora excluding `0x00`;
- independent shard indexes;
- shard manifests;
- fan-out and deterministic shard merge;
- logical multi-shard prototypes.

That architecture is historical.

Its documents and implementation paths are retained under legacy and
experimental areas for evidence, comparison and regression analysis.

Important legacy limitation:

A byte sequence crossing an arbitrary physical shard boundary may be missed
when independently indexed shards are treated as one logical byte stream.

The current verified multi-document model avoids this ambiguity by defining
documents as independent matching domains.

Legacy segmented behavior must not be treated as authoritative for the
binary-safe runtime or for future composition semantics.

## Composition research direction

A future composition layer may combine multiple already verified runtime
units.

Possible concerns include:

- committed composition identity;
- stable global document identity;
- deterministic fan-out;
- canonical result merge;
- complete coverage evidence;
- fail-closed missing-unit behavior;
- independent composition replay.

This layer is not currently implemented or verified.

No current required verification claim includes an Index Forest,
field-aware routing or multi-block composition.

Any future composition work must be defined above the verified runtime and
operator layers without weakening P1-P12, R0-R6 or O1-O6.

## Experimental and research paths

The repository contains additional work including:

- compressed RLBWT experiments;
- compact runtime and locate research;
- segmented and persistent-query prototypes;
- HTTP experiments;
- Structural Fingerprint replay experiments;
- public benchmark runbooks;
- future composition research;
- future hardware-acceleration research.

These paths do not automatically belong to the required verification chain.

A research artifact becomes part of the verified architecture only after its
claims, executable gates, mutation tests and required closure are committed.

## Architectural rule

GLYPH development follows this ordering:

    semantics
    -> executable reference oracle
    -> compiled implementation
    -> differential verification
    -> evidence and replay
    -> optimization
    -> public claim

Optimization must not redefine correctness.

A newer architecture may supersede an older implementation path, but the
historical evidence remains preserved.
