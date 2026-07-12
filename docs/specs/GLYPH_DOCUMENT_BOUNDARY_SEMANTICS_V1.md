# GLYPH_DOCUMENT_BOUNDARY_SEMANTICS_V1

Status: normative draft  
Version: 1  
Proof obligation: P9  
Date: 2026-07-11

## Purpose

Define document-boundary semantics for GLYPH multi-document corpora.

P9 proves that a query match is valid only when all query bytes belong to one
document.

A byte sequence formed by concatenating the suffix of one document with the
prefix of another document is not a valid match.

## Dependencies

P9 depends on proof obligations P1 through P8.

## Canonical corpus model

A corpus is an ordered list:

    documents = [D0, D1, ..., Dn-1]

Each document is an independent byte sequence.

Document identity and order are committed under P4.

The corpus is not semantically equivalent to:

    D0 || D1 || ... || Dn-1

for substring matching.

## Valid match

For non-empty query `q`, coordinate `(doc_id, doc_offset)` is a valid match if
and only if:

    0 <= doc_id < document_count

    0 <= doc_offset

    doc_offset + len(q) <= len(document[doc_id])

    document[doc_id][doc_offset:doc_offset+len(q)] == q

All bytes must come from the same document.

## Invalid cross-document match

Given:

    D0 = b"ab"
    D1 = b"cd"

query:

    b"bc"

is not present.

The byte sequence exists only in the accidental physical concatenation:

    b"abcd"

It is not part of either document.

Required result:

    match_count = 0
    coordinates = []

## Count semantics

Count must equal the number of document-local coordinates.

Required:

    FM_count
    ==
    naive_document_local_count
    ==
    len(full_locate_coordinates)

Count must not include suffixes whose query span crosses a document boundary.

Boundary filtering only after count is forbidden.

## Locate semantics

Locate must return only canonical coordinates:

    (doc_id, doc_offset)

Every returned span must fit completely inside the referenced document.

A global concatenated offset is insufficient unless it is deterministically
converted and checked against the document boundary table.

## Artifact semantics

An evidence artifact for a multi-document corpus must bind:

- corpus identity;
- document table identity;
- query identity;
- exact match count;
- canonical document-local coordinates;
- document-boundary policy identifier.

Required policy identifier:

    document_boundary_policy =
        "DOCUMENT_LOCAL_MATCHES_ONLY_V1"

## Replay semantics

Replay must independently verify:

1. corpus identity;
2. document ordering;
3. document lengths;
4. query identity;
5. every coordinate;
6. every matched byte span;
7. exact count;
8. absence of cross-document coordinates;
9. boundary policy identifier.

Replay must not trust `byte_check=true` without recomputing the byte spans.

## Physical representation

An implementation may use:

- virtual sentinels;
- boundary tables;
- separate indexes;
- a generalized suffix array;
- another representation preserving the same semantics.

The physical implementation is not authoritative.

The document-local match model is authoritative.

## Boundary table

For a physical concatenation, a boundary table must at minimum determine:

- document ID;
- global start offset;
- document byte length;
- global exclusive end offset.

A global candidate `[start, start + query_length)` is valid only if:

    start >= document_start
    start + query_length <= document_end

for one and the same document.

## Empty documents

Empty documents:

- remain part of corpus identity;
- contribute no suffix byte positions;
- produce no query matches;
- preserve their document IDs.

## Duplicate documents

Identical documents remain distinct.

Example:

    D0 = b"same"
    D1 = b"same"

query:

    b"same"

produces:

    match_count = 2
    coordinates = [(0, 0), (1, 0)]

## Prefix-related documents

Given:

    D0 = b"a"
    D1 = b"ab"
    D2 = b"abc"

query `b"ab"` matches only:

    (1, 0)
    (2, 0)

It must not be synthesized from the end of D0 and the beginning of D1.

## Binary boundaries

Boundary semantics apply identically when documents begin or end with:

- `0x00`;
- `0xFF`;
- invalid UTF-8 bytes;
- newline;
- carriage return;
- repeated binary values.

## Queries ending at document boundary

A query may end exactly at a document boundary.

Valid condition:

    doc_offset + query_length == document_length

It must not consume any byte from the next document.

## Queries longer than a document

A query longer than a document cannot match inside that document.

No continuation into the next document is permitted.

## Bounded locate

Bounded locate must apply only after the exact document-local result set is
known.

Required:

    match_count = exact full document-local count

    returned_coordinates =
        canonical_prefix(full_document_local_coordinates)

A bounded locate response must not hide a cross-document count discrepancy.

## Required fixtures

P9 fixtures must include:

1. `b"ab"` + `b"cd"` queried with `b"bc"`;
2. `b"\x00"` + `b"\xff"` queried with `b"\x00\xff"`;
3. `b"\xff"` + `b"\x00"` queried with `b"\xff\x00"`;
4. empty document between non-empty documents;
5. duplicate documents;
6. prefix-related documents;
7. one-byte documents;
8. query ending exactly at document end;
9. query longer than every document;
10. all 256 byte values split across documents;
11. repeated same byte split across boundaries;
12. newline and carriage-return boundaries.

## Required mutation failures

The validator must reject:

1. count includes a cross-document occurrence;
2. locate includes a cross-document coordinate;
3. global offset maps across a boundary;
4. coordinate uses wrong document;
5. coordinate starts at document end;
6. coordinate ends after document end;
7. artifact omits boundary policy;
8. artifact uses wrong boundary policy;
9. artifact count differs from document-local oracle;
10. replay trusts stored `byte_check=true`;
11. replay uses physical concatenation as oracle;
12. duplicate document occurrence is collapsed;
13. empty document changes later document IDs;
14. document order mutation is ignored;
15. document length mutation is ignored;
16. bounded locate prefixes an invalid full result.

## P9 invariant

For every non-empty query:

    result_set
    =
    {
      (doc_id, doc_offset)
      |
      document[doc_id][doc_offset:doc_offset+len(query)]
      == query
      and
      doc_offset + len(query) <= len(document[doc_id])
    }

No other occurrence exists in GLYPH semantics.

## Non-claims

P9 does not yet prove:

- artifact schema integrity;
- bundle manifest integrity;
- portable replay path integrity;
- tamper-evident packaging;
- membership or non-membership proof completeness.

Those belong to later obligations.

## Completion condition

P9 is complete only when:

1. this specification exists;
2. an independent document-local oracle exists;
3. count equals document-local count;
4. locate equals document-local coordinates;
5. artifact carries the boundary policy;
6. replay independently recomputes boundaries;
7. binary boundary fixtures pass;
8. all mutation fixtures fail;
9. top-level verification remains green.
