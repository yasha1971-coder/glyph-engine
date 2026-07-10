# GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1

Status: normative draft  
Version: 1  
Proof obligation: P1  
Date: 2026-07-11

## Purpose

Define the canonical total order used by future binary-safe GLYPH suffix-array,
BWT, FM-index, RLBWT, locate, fingerprint, and replay implementations.

This specification is normative.

Implementations and third-party libraries must conform to this order.
Library-specific ordering behavior is not accepted as the GLYPH definition.

## Logical corpus model

A logical corpus is an ordered sequence of documents:

    C = [D_0, D_1, ..., D_(k-1)]

Each document is an arbitrary finite byte sequence.

A document may contain every byte value:

    0x00 .. 0xFF

The physical byte 0x00 is ordinary corpus data.

It is not a terminator, separator, sentinel, or end-of-string marker.

## Document-local suffixes

For document D_d of length n_d, byte-start suffixes are:

    S(d, i) = D_d[i : n_d]

where:

    0 <= i < n_d

Suffixes never continue into another document.

No searchable suffix crosses a document boundary.

Empty documents contain no byte-start suffixes.

## Virtual end-of-document symbol

Every document has a conceptual virtual end marker:

    END(d)

The marker is not a byte.

It cannot be supplied in a query.

It cannot be serialized as a corpus byte.

It cannot match any byte value, including 0x00.

The conceptual terminated suffix is:

    S*(d, i) = D_d[i : n_d] || END(d)

## Symbol order

For every document d and every byte b:

    END(d) < b

For bytes:

    0x00 < 0x01 < ... < 0xFF

For virtual document ends:

    END(d1) < END(d2)  iff  d1 < d2

Therefore the complete conceptual order is:

    END(0) < END(1) < ... < END(k-1)
           < 0x00 < 0x01 < ... < 0xFF

The document order is the manifest order.

## Suffix comparison

To compare S*(d1, i1) and S*(d2, i2):

1. Compare successive corpus bytes numerically as unsigned values.
2. At the first unequal byte, the smaller byte wins.
3. If one document ends while the other still has bytes, the ended suffix wins,
   because END(d) is smaller than every byte.
4. If both suffixes end at the same comparison depth, the suffix from the
   lower document ID wins.

Equivalent rule:

    lexicographic unsigned-byte order
    + shorter suffix first
    + document ID tie-break

## Required examples

Single-document examples:

    D = "aa"

    suffix at 1: "a"
    suffix at 0: "aa"

    SA order: [(0,1), (0,0)]

    D = 00 00

    suffix at 1: 00
    suffix at 0: 00 00

    SA order: [(0,1), (0,0)]

    D = FF 00

    suffix at 1: 00
    suffix at 0: FF 00

    SA order: [(0,1), (0,0)]

Duplicate-document example:

    D_0 = "a"
    D_1 = "a"

    S*(0,0) = "a" END(0)
    S*(1,0) = "a" END(1)

    Since END(0) < END(1):

    SA order: [(0,0), (1,0)]

Prefix example:

    D_0 = "ab"
    D_1 = "abc"

    "ab" END(0) < "abc" END(1)

Cross-document concatenation is not part of comparison semantics.

## Query alphabet

Queries are finite byte sequences over:

    {0x00 .. 0xFF}

Virtual end markers are outside the query alphabet.

A query can never directly or indirectly match END(d).

## Determinism requirement

Given identical:

- ordered document manifest
- document byte lengths
- document bytes
- format version

all conforming implementations must produce the same ordered list of
document-local suffix coordinates:

    (doc_id, doc_offset)

The order must not depend on:

- thread scheduling
- unstable sorting
- operating system
- locale
- host endianness
- compiler
- suffix-array library
- incidental input allocation order

## Coordinate model

Canonical suffix coordinates are:

    (doc_id, doc_offset)

Global concatenation offsets are not normative.

They may be derived for presentation, but must not define suffix identity,
evidence identity, or replay semantics.

## Explicit non-rules

GLYPH V1 does not define suffix order using:

- C-string termination
- physical 0x00 sentinels
- locale collation
- signed char comparison
- rotation order
- arbitrary library tie order
- concatenation across document boundaries

## P1 invariant

For every two distinct byte-start suffix coordinates A and B:

    exactly one of A < B or B < A is true

and the result is determined solely by this specification.

## Completion condition

P1 is complete only when:

1. this specification exists;
2. an independent executable oracle implements it;
3. required fixtures pass;
4. duplicate builds produce identical ordered coordinates.
