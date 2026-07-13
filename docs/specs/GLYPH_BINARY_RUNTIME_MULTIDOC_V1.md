# GLYPH_BINARY_RUNTIME_MULTIDOC_V1

Status: implementation gate  
Version: 1  
Date: 2026-07-13

## Purpose

Define the first binary-safe multi-document runtime path for GLYPH.

## Architecture

Each document is indexed independently with:

- `build_sa_binary_v1`
- `build_bwt_binary_v1`
- `build_fm_binary_v1`
- `query_fm_locate_binary_v1`

A multi-document corpus is therefore an ordered collection of independent
single-document indexes.

The runtime does not physically concatenate source documents.

## Consequence

A byte query cannot cross from one document into another because no FM index
contains bytes from more than one document.

Cross-document matches are structurally impossible rather than filtered after
search.

## Coordinate model

Results use canonical coordinates:

    (document_id, document_offset)

Documents retain their position in the ordered document sequence, including
empty documents.

## Count semantics

The multi-document match count is:

    sum(document-local match counts)

## Locate semantics

All document-local coordinates are merged and ordered by:

    document_id ascending
    document_offset ascending

Bounded locate returns the canonical prefix of this complete coordinate set.

## Duplicate documents

Byte-identical documents remain separate corpus members and produce separate
coordinates with distinct document IDs.

## Empty documents

Empty documents contribute no byte matches but retain their document IDs.

## Binary domain

Every document may contain arbitrary bytes:

    0x00 through 0xFF

The virtual sentinel remains internal to each independent index and cannot be
returned as a byte match.

## Non-claims

This gate does not yet establish:

- committed multi-document corpus identity;
- deterministic evidence artifact replay;
- self-contained runtime bundle;
- full runtime conformance.

Those belong to the next runtime artifact and bundle layers.
