# GLYPH_SUFFIX_ARRAY_VALIDITY_V1

Status: normative draft  
Version: 1  
Proof obligation: P2  
Date: 2026-07-11

## Purpose

Define the validity conditions for a binary-safe, multi-document GLYPH suffix array.

This specification depends on:

    GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1

A suffix array is valid only if it is:

1. a complete permutation of all document-local byte-start suffixes;
2. free of duplicates;
3. free of invalid coordinates;
4. strictly increasing under the canonical P1 suffix order.

## Corpus model

A logical corpus is an ordered sequence of documents:

    C = [D_0, D_1, ..., D_(k-1)]

Each document is an arbitrary finite byte sequence over:

    0x00 .. 0xFF

For document `D_d` of length `n_d`, valid suffix coordinates are:

    (d, i)

where:

    0 <= d < k
    0 <= i < n_d

Empty documents contribute no suffix coordinates.

## Expected suffix set

The complete expected suffix-coordinate set is:

    U(C) = {
        (d, i)
        |
        0 <= d < document_count
        and
        0 <= i < len(D_d)
    }

Its size is:

    N = sum(len(D_d))

## Candidate suffix array

A candidate suffix array is an ordered list:

    SA = [SA[0], SA[1], ..., SA[N-1]]

where each entry is a canonical coordinate:

    (doc_id, doc_offset)

Global concatenation offsets are not normative suffix identities.

## Validity requirements

### V1 — Exact length

A valid SA must contain exactly:

    N = sum(len(D_d))

entries.

### V2 — Coordinate validity

Every coordinate `(d, i)` must satisfy:

    0 <= d < document_count
    0 <= i < len(D_d)

Coordinates into empty documents are invalid.

Negative offsets are invalid.

Offsets equal to document length are invalid.

### V3 — Uniqueness

No coordinate may appear more than once.

### V4 — Completeness

Every coordinate in `U(C)` must appear exactly once.

Therefore:

    set(SA) = U(C)

### V5 — Strict canonical order

For every adjacent pair:

    SA[j] < SA[j + 1]

under:

    GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1

Equality is forbidden.

Descending adjacent pairs are forbidden.

Library-specific suffix order is not accepted when it differs from P1.

### V6 — Determinism

For identical:

- ordered document manifest;
- document bytes;
- document lengths;
- format version;

the valid canonical suffix array must be identical.

The result must not depend on:

- thread scheduling;
- host endianness;
- signed `char`;
- locale;
- unstable sorting;
- allocation order;
- suffix-array library tie behavior.

## Required rejection classes

A conforming validator must reject:

1. a missing suffix coordinate;
2. a duplicate coordinate;
3. an invalid document ID;
4. a negative document offset;
5. an offset equal to document length;
6. an offset larger than document length;
7. a coordinate inside an empty document;
8. an adjacent inversion;
9. a non-adjacent permutation that is not sorted;
10. a candidate using global offsets instead of `(doc_id, doc_offset)`.

## Required positive fixtures

At minimum:

- empty corpus;
- one empty document;
- `aa`;
- `00 00`;
- `FF 00`;
- duplicate documents;
- prefix-related documents;
- periodic bytes;
- full byte alphabet;
- duplicate binary documents;
- empty documents mixed with non-empty documents.

## P2 invariant

For a valid candidate SA:

    len(SA) = |U(C)|

and:

    set(SA) = U(C)

and for all valid adjacent positions `j`:

    compare_P1(SA[j], SA[j + 1]) < 0

## Relationship to implementation

This specification validates the semantic output of an SA builder.

It does not prescribe:

- SA-IS;
- prefix doubling;
- induced sorting;
- libsais;
- internal integer remapping;
- physical corpus layout.

Any implementation is acceptable only when its decoded coordinate output passes P2.

## Non-claims

P2 does not prove:

- BWT correctness;
- LF-mapping correctness;
- FM-index count correctness;
- locate correctness;
- document-boundary filtering;
- artifact replay correctness.

Those are separate proof obligations.

## Completion condition

P2 is complete only when:

1. this specification exists;
2. an independent executable validator exists;
3. positive fixtures pass;
4. required mutation fixtures fail;
5. existing GLYPH verification remains green.

## Terminal suffix row correction

Normative correction:

For every document of byte length `n`, the canonical generalized suffix
matrix contains coordinates:

    (doc_id, 0)
    ...
    (doc_id, n)

The coordinate `(doc_id, n)` is the terminal suffix consisting only of that
document's virtual sentinel.

Therefore every document contributes:

    n + 1 suffix rows

including an empty document, which contributes one virtual-sentinel-only row.

This terminal row is required so that:

- the first and last FM columns have the same token multiset;
- every source byte has a predecessor relation in BWT;
- every virtual sentinel occurs exactly once;
- LF is a permutation;
- backward search intervals correspond to contiguous suffix-array rows.

A suffix array containing only offsets `0..n-1` is insufficient for the
virtual-sentinel FM construction and must be rejected.
