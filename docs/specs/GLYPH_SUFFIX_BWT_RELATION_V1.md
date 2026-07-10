# GLYPH_SUFFIX_BWT_RELATION_V1

Status: normative draft  
Version: 1  
Proof obligation: P3  
Date: 2026-07-11

## Purpose

Define the canonical relationship between:

- an ordered multi-document byte corpus;
- a valid canonical suffix array;
- the corresponding GLYPH suffix-BWT token sequence.

This specification exists to prevent accidental substitution of:

- cyclic rotation BWT;
- wrap-around predecessor bytes;
- physical byte sentinels;
- ambiguous document separators;
- library-specific BWT conventions.

## Dependencies

This specification depends on:

- `GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1`
- `GLYPH_SUFFIX_ARRAY_VALIDITY_V1`

P3 assumes that the input suffix array already satisfies P2.

## Corpus model

A logical corpus is an ordered sequence of documents:

    C = [D_0, D_1, ..., D_(k-1)]

Each document is an arbitrary finite byte sequence over:

    0x00 .. 0xFF

Document identity is part of the corpus model.

Documents are not joined by physical separator bytes.

## Suffix coordinates

A suffix coordinate is:

    (doc_id, doc_offset)

with:

    0 <= doc_id < document_count
    0 <= doc_offset < len(D_doc_id)

Empty documents contribute no suffix coordinates and therefore no BWT rows.

## Valid suffix array

The suffix array:

    SA = [SA[0], SA[1], ..., SA[N-1]]

must be a valid P2 suffix array over all byte-start suffix coordinates.

Its length is:

    N = sum(len(D_d))

## BWT token alphabet

The canonical BWT is not necessarily a raw byte string.

Each BWT position contains one tagged token from:

    BYTE(value)

where:

    0 <= value <= 255

or:

    VIRTUAL_SENTINEL(doc_id)

A virtual sentinel is not a byte.

In particular:

    VIRTUAL_SENTINEL(doc_id) != BYTE(0x00)

and:

    VIRTUAL_SENTINEL(doc_id) != BYTE(0xFF)

No byte value may be reserved as an implicit sentinel.

## Canonical SA-to-BWT mapping

For each suffix-array row `j`:

    SA[j] = (d, i)

define:

    BWT[j] =
        BYTE(D_d[i - 1])              if i > 0
        VIRTUAL_SENTINEL(d)           if i = 0

This mapping is total for every valid P2 suffix coordinate.

## Start-of-document rule

When:

    doc_offset = 0

the predecessor is:

    VIRTUAL_SENTINEL(doc_id)

It is not:

- the final byte of the same document;
- the final byte of the previous document;
- a physical zero byte;
- a shared untyped `$`;
- a host-language string terminator.

## No wrap-around rule

GLYPH suffix-BWT is not cyclic rotation BWT.

For a document:

    D = b_0 b_1 ... b_(n-1)

the suffix beginning at offset zero has predecessor:

    VIRTUAL_SENTINEL(doc_id)

not:

    b_(n-1)

Therefore the following implementation is invalid:

    predecessor = D[(offset - 1) mod len(D)]

The modulo operation creates cyclic rotation semantics and violates P3.

## Document isolation

The predecessor of a suffix is always resolved inside the same logical document.

For suffix coordinate:

    (d, i)

P3 never reads bytes from:

    D_(d-1)
    D_(d+1)

Physical concatenation layout is non-normative.

A suffix at offset zero must not inherit a byte from the preceding physically concatenated document.

## Virtual sentinel identity

The token:

    VIRTUAL_SENTINEL(d)

preserves document identity.

Implementations must not collapse all document sentinels into an untyped byte before semantic validation.

An implementation may use an internal representation, but the decoded semantic token must retain the correct `doc_id`.

## Canonical token encoding

For artifacts and fixtures, the canonical JSON representation is:

    {
      "kind": "byte",
      "value": 0
    }

through:

    {
      "kind": "byte",
      "value": 255
    }

or:

    {
      "kind": "virtual_sentinel",
      "doc_id": 0
    }

The two token kinds are disjoint.

## BWT length invariant

For a valid P2 suffix array:

    len(BWT) = len(SA)

and:

    len(BWT) = sum(len(D_d))

Every SA row produces exactly one BWT token.

## Sentinel multiplicity invariant

Each non-empty document contributes exactly one start suffix:

    (doc_id, 0)

Therefore each non-empty document contributes exactly one token:

    VIRTUAL_SENTINEL(doc_id)

Empty documents contribute zero BWT tokens.

For each document `d`:

    count(BWT, VIRTUAL_SENTINEL(d)) =
        1 if len(D_d) > 0
        0 if len(D_d) = 0

## Byte multiplicity observation

For a non-empty document:

- bytes at offsets `0 .. n-2` appear as predecessor byte tokens;
- the final byte at offset `n-1` does not appear through wrap-around;
- one virtual sentinel appears instead.

P3 does not claim that the BWT byte histogram equals the source byte histogram.

Such equality would generally indicate cyclic wrap-around semantics.

## Required rejection classes

A conforming P3 validator must reject:

1. BWT length different from SA length;
2. a raw byte used instead of a virtual sentinel at document start;
3. `BYTE(0x00)` used as a sentinel;
4. final-document-byte wrap-around at offset zero;
5. predecessor byte taken from a previous physical document;
6. wrong virtual sentinel document ID;
7. sentinel attached to a suffix whose offset is non-zero;
8. wrong predecessor byte for a non-zero suffix offset;
9. collapsed shared sentinel identity;
10. malformed or unknown token kinds.

## Required fixtures

Positive fixtures must include:

- empty corpus;
- empty document;
- one-byte document;
- real `0x00`;
- real `0xFF`;
- `0x00 0xFF`;
- duplicate documents;
- prefix-related documents;
- periodic data;
- multiple documents;
- empty documents mixed with non-empty documents;
- full byte alphabet.

Mutation fixtures must include:

- rotation-BWT substitution;
- zero-byte sentinel substitution;
- previous-document predecessor substitution;
- wrong document-local sentinel;
- byte-token corruption;
- missing token;
- extra token;
- malformed token.

## Rotation-BWT distinction fixture

For:

    D_0 = "aba"

the suffix at:

    (0, 0)

must contribute:

    VIRTUAL_SENTINEL(0)

A cyclic implementation would contribute:

    BYTE("a")

The two outputs are semantically different.

The cyclic result must be rejected.

## Determinism

For identical:

- ordered document manifest;
- source bytes;
- P1 ordering;
- P2 suffix array;
- format version;

the canonical BWT token sequence must be identical.

The output must not depend on:

- host endianness;
- signed `char`;
- locale;
- physical concatenation;
- allocator behavior;
- thread scheduling;
- suffix-array library BWT mode.

## Relationship to raw BWT files

Existing raw-byte BWT files may only represent P3 when their format defines an unambiguous encoding for virtual sentinel tokens.

A byte-only file without an external sentinel map cannot represent all 256 byte values plus document-local virtual sentinel identities without additional metadata.

Therefore:

    raw byte BWT alone is not sufficient evidence of P3 conformance

for fully binary-safe multi-document corpora.

## P3 invariant

For every valid SA row:

    SA[j] = (d, i)

the canonical BWT token must satisfy:

    i > 0
        => BWT[j] = BYTE(D_d[i - 1])

    i = 0
        => BWT[j] = VIRTUAL_SENTINEL(d)

No modulo wrap-around is permitted.

## Non-claims

P3 does not prove:

- invertibility;
- LF-mapping correctness;
- C-array correctness;
- rank correctness;
- backward-search correctness;
- count correctness;
- locate correctness;
- document-boundary match filtering;
- portable artifact replay.

Those are separate proof obligations.

## Completion condition

P3 is complete only when:

1. this specification exists;
2. an independent executable oracle exists;
3. positive fixtures pass;
4. rotation-BWT mutations fail;
5. byte/sentinel alias mutations fail;
6. document-crossing predecessor mutations fail;
7. existing top-level verification remains green.
