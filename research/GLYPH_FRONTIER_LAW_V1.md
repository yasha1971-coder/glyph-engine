# GLYPH_FRONTIER_LAW_V1

Status: Research note  
Date: 2026-06-26

## Purpose

This note records the corrected frontier for GLYPH after the zero-rescan positioning review.

GLYPH is not currently validated as a standalone product.

GLYPH is a research artifact for measuring when exact-byte evidence over fixed corpora becomes physically valuable.

## Breakthrough Law

A real technical breakthrough usually appears when three things meet:

1. An existing capability
2. A new context
3. A previously under-measured metric

For GLYPH:

Existing capability:

- deterministic exact-byte retrieval
- suffix-array / BWT / FM-index based search
- offsets
- byte-checks
- replayable audit artifacts

New context:

- fixed corpora
- large or remote archives
- immutable evidence objects
- forensic/log/source/package/data snapshots
- cases where a second party must verify exact byte presence

Previously under-measured metric:

- verifier-side cost
- rescan cost
- proof size
- offset validation cost
- trust boundary between finder and verifier
- cost of turning a search result into a portable evidence object

## What was rejected

The following framings are currently too weak or misleading as public lead claims:

- GLYPH as a general search engine
- GLYPH as a SIEM/ELK/Splunk replacement
- GLYPH as a legal proof system
- GLYPH as a semantic truth system
- GLYPH as a COBOL migration verifier
- GLYPH as proof of an entire real-world incident
- GLYPH as a verifier-side requirement when offset + seek is enough

## Critical physical objection

If a finder already gives a verifier:

- corpus hash
- offset
- length
- expected bytes or query bytes

and the verifier already has the same corpus, then the verifier does not need GLYPH.

The verifier can use a standard seek/read operation.

Example:

    dd if=corpus.raw bs=1 skip=<offset> count=<length>

or an equivalent `seek()` call.

Therefore, the value of GLYPH cannot be merely:

"the verifier does not rescan."

That collapses to:

"the verifier has offsets."

This is not enough to justify GLYPH as a standalone product.

## Corrected frontier

The real question is narrower:

When does a search result become valuable as a standardized, replayable, portable, independently checkable evidence receipt rather than just an offset list?

This is not proven.

It must be measured.

## GLYPH's real research object

GLYPH should be treated as a tool for studying this boundary:

    search result
    → coordinate
    → receipt
    → replay
    → third-party verification

The core research question:

At what corpus size, remoteness, audit pressure, or trust boundary does a portable exact-byte evidence receipt become worth more than an ordinary offset file?

## Measurement axes

Future GLYPH work should only continue if it measures at least one of these:

1. Rescan cost

How expensive is it to re-run search over the corpus?

Metrics:

- corpus size
- scan time
- I/O cost
- cloud egress cost
- cold storage retrieval cost

2. Verification cost

How expensive is it to verify a reported match?

Metrics:

- seek/read time
- required bytes read
- number of offsets
- verifier memory
- verifier tooling complexity

3. Receipt value

What does the artifact add beyond offset + length?

Possible components:

- corpus identity
- query identity
- match count
- offsets
- snippets
- byte-checks
- replay command
- tool version
- manifest hash
- source manifest
- reproducibility boundary

4. Trust boundary

Who does not trust whom?

Examples:

- analyst → auditor
- vendor → customer
- data holder → regulator
- model builder → dataset auditor
- incident responder → legal/compliance team

5. Completeness boundary

Does the recipient need to know only that one match exists, or that all matches were reported?

This distinction is crucial.

Existence proof is easy with offsets.

Completeness proof is hard.

GLYPH is currently closer to reproducible retrieval than to cryptographic completeness.

## Existing GLYPH status

Implemented / demonstrated:

- exact byte retrieval over fixed corpora
- FM interval
- match count
- offsets when locate layer exists
- byte-checks
- Audit Artifact V0
- Evidence Case V1
- replay verification
- mini demo
- Pizza 50MB public-style demo
- XZ CVE Phase 1 experimental demo

Not implemented:

- authenticated FM-index
- cryptographic completeness proof
- non-membership proof
- verifier without corpus
- Merkleized BWT/rank proof
- SNARK/STARK proof
- compact universal inclusion proof
- production r-index
- production PFP builder
- commercial-grade compressed index

## Ultra-GLYPH is not a claim

Ideas such as:

- r-index
- H_k-compressed indexes
- prefix-free parsing
- compressed BWT
- Merkle commitments
- authenticated rank/select
- compact inclusion receipts

are research directions.

They must not be presented as current GLYPH capabilities.

## Possible future architecture

A future GLYPH-like system could separate into three layers:

1. Finder layer

Responsible for search.

Possible technologies:

- FM-index
- r-index
- compressed suffix structures
- PFP-built BWT
- sampled locate

2. Receipt layer

Responsible for packaging evidence.

Contains:

- corpus commitment
- query commitment
- match count
- offsets
- snippets or snippet hashes
- byte-check records
- replay command
- tool identity

3. Verifier layer

Responsible for checking the receipt.

Possible modes:

- seek/read verifier using the original corpus
- Merkle chunk verifier
- authenticated index verifier
- future cryptographic proof verifier

Current GLYPH implements mainly layer 1 and a prototype of layer 2.

It does not yet implement a true layer 3 cryptographic proof system.

## The hard fork in the project

There are only two serious futures for GLYPH:

### Path A — Freeze

Freeze GLYPH as a technically valid research artifact.

Condition:

No external user confirms a real need for portable exact-byte evidence receipts beyond offset + seek.

Result:

GLYPH remains useful as:

- research record
- reproducibility demo
- exact-byte retrieval prototype
- evidence-chain experiment

### Path B — Convert to Receipt Protocol

Stop treating GLYPH as a search engine.

Turn it into a minimal standard for exact-byte evidence receipts.

Core object:

    corpus_hash
    query_hash
    offsets
    lengths
    expected_bytes_hash
    byte_check
    source_manifest
    replay_command
    tool_identity

Goal:

Make the receipt format useful even when the underlying search engine is grep, ripgrep, ClickHouse, GLYPH, or another index.

This would mean GLYPH's durable asset is not the index.

The durable asset is the evidence receipt protocol.

## Decision rule

Do not continue GLYPH development unless one of these happens:

1. A real external user provides a fixed corpus and says offset + seek is not enough.
2. A real workflow requires standardized receipts across teams/tools.
3. A real audit/compliance setting requires reproducible exact-byte evidence.
4. A technical path emerges for compact authenticated inclusion/completeness proofs.
5. GLYPH is reused internally as a component of another system.

Otherwise:

Freeze active development.

## Current conclusion

GLYPH works.

But working is not enough.

The current product hypothesis is unvalidated.

The zero-rescan framing alone collapses to offset + seek.

The stronger frontier is:

    exact-byte evidence receipt protocol
    over fixed corpora
    with measured verifier-side value

Until that value is externally confirmed, GLYPH should be treated as a frozen research layer, not an active product.
