# GLYPH_REPLAY_DETERMINISM_V1

Status: normative draft  
Version: 1  
Proof obligation: P10  
Date: 2026-07-12

## Purpose

Define deterministic generation, serialization, identity, and replay rules for
GLYPH evidence artifacts.

## Dependencies

P10 depends on P1 through P9.

## Core invariant

For identical authoritative inputs:

- corpus identity;
- ordered document table;
- query bytes;
- format version;
- boundary policy;
- algorithm parameters;
- locate bound;

GLYPH must produce identical:

- authoritative artifact object;
- canonical JSON bytes;
- artifact SHA256;
- replay result.

## Authoritative fields

Authoritative fields include:

- artifact version;
- corpus identity;
- document count and lengths;
- document-boundary policy;
- query hex, length, and SHA256;
- exact match count;
- canonical coordinates;
- bounded-locate parameters and flags;
- byte-check result;
- semantic/profile identifiers.

## Observational fields

The following must not participate in artifact identity:

- timestamps;
- hostname;
- process ID;
- current working directory;
- temporary paths;
- absolute source paths;
- duration;
- random nonce;
- log text.

They may exist only outside the authoritative payload.

## Canonical JSON

Canonical authoritative JSON uses:

- UTF-8;
- lexicographically sorted object keys;
- compact separators `,` and `:`;
- no insignificant whitespace;
- deterministic escaping;
- no NaN or Infinity;
- deterministic list ordering.

## Artifact digest

The artifact digest is:

    SHA256(canonical_authoritative_json_bytes)

The digest field itself is outside the digest preimage.

## Coordinate order

Coordinates are ordered by:

    doc_id ascending
    doc_offset ascending

Duplicate coordinates are forbidden.

## Replay

Replay must:

1. validate all authoritative fields;
2. independently decode query bytes;
3. independently recompute corpus identity;
4. independently recompute document-local matches;
5. independently recompute bounded output;
6. independently canonicalize the artifact;
7. independently recompute its SHA256;
8. reject every mismatch.

## Required deterministic fixtures

- ASCII;
- embedded `0x00`;
- `0xFF`;
- invalid UTF-8;
- empty documents;
- duplicate documents;
- zero matches;
- bounded locate;
- complete byte alphabet `00..ff`;
- repeated runs.

Each fixture must be generated at least three times.

## Required mutation failures

The validator must reject:

- corpus identity mutation;
- query mutation;
- query-length mutation;
- query-hash mutation;
- match-count mutation;
- coordinate mutation;
- unsorted coordinates;
- duplicate coordinates;
- boundary-policy mutation;
- artifact-version mutation;
- digest mutation;
- observational field inside authoritative payload;
- changed document order;
- changed document bytes;
- false byte-check;
- changed bounded-locate parameters.

## P10 invariant

For valid repeated generations `A`, `B`, and `C`:

    canonical(A) == canonical(B) == canonical(C)

and:

    SHA256(canonical(A))
    ==
    SHA256(canonical(B))
    ==
    SHA256(canonical(C))

## Non-claims

P10 does not yet prove:

- bundle completeness;
- portable bundle replay;
- complete manifest coverage;
- end-to-end source-to-bundle closure.

Those belong to P11 and P12.

## Completion condition

P10 is complete when:

- this specification exists;
- validator fixtures pass;
- repeated generation is byte-identical;
- reordered JSON keys canonicalize identically;
- observational metadata cannot affect identity;
- all mutations are rejected;
- existing `./verify.sh` remains green.
