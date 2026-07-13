# GLYPH_BINARY_RUNTIME_EVIDENCE_V1

Status: implementation gate  
Version: 1  
Date: 2026-07-13

## Purpose

Define a deterministic evidence artifact produced by the actual compiled
GLYPH binary-safe C++ runtime.

The artifact binds:

- the ordered source-document corpus;
- each source document's byte length and SHA256;
- deterministic SA, BWT, and FM runtime files for every document;
- the exact binary query;
- the C++ count and locate result;
- canonical document-local coordinates;
- bounded-locate metadata;
- the runtime profile and binary-format version.

## Runtime profile

    GLYPH_BINARY_RUNTIME_V1

## Index topology

    one independent index per document

Documents are never physically concatenated.

## Corpus identity

The ordered corpus identity preimage is:

    ASCII("GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1")
    0x00
    document_count_u64_be
    repeated for each document in document-id order:
        document_id_u64_be
        document_length_u64_be
        document_sha256_raw_32_bytes

The resulting corpus identity is SHA256 of this preimage.

Document order, empty documents, and duplicate documents are authoritative.

## Query identity

The authoritative query representation is canonical lowercase hexadecimal.

The artifact binds:

- `query_hex`;
- `query_length_bytes`;
- `query_sha256`.

The SHA256 preimage is the decoded query byte sequence.

## Per-document runtime commitment

For every document, the artifact records deterministic commitments for:

- `GLYPH_SA_BINARY_V1`;
- `GLYPH_BWT_BINARY_V1`;
- `GLYPH_FM_BINARY_V1`.

Each commitment contains:

- format name;
- byte size;
- SHA256.

Replay rebuilds these files using the compiled C++ runtime and requires exact
hash equality.

## Result semantics

The artifact records:

- complete FM-derived match count;
- canonical `(document_id, document_offset)` coordinates;
- returned count;
- bounded status;
- offsets-complete status;
- byte-check status.

For bounded locate, coordinates are the canonical prefix of the complete
coordinate sequence.

## Replay

Replay must:

1. validate artifact structure and constants;
2. verify source document hashes and ordered corpus identity;
3. verify query encoding, length, and SHA256;
4. rebuild every per-document C++ index;
5. compare rebuilt index hashes to the artifact;
6. rerun C++ count and locate;
7. recompute canonical global coordinates;
8. compare all authoritative result fields;
9. byte-check every returned coordinate.

Replay must reject any mismatch.

## Determinism

Two builds over identical source bytes, document order, query, runtime version,
and bounded-locate parameter must produce byte-identical canonical JSON.

## Non-claims

This artifact does not yet establish:

- a self-contained portable runtime bundle;
- replay without separately supplied source documents;
- integration into the top-level proof graph;
- complete GLYPH runtime conformance.

Those are subsequent gates.
