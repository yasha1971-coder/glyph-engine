# GLYPH_FM_TOKEN_RANK_LF_CONSISTENCY_V1

Status: normative draft  
Version: 1  
Proof obligation: P5  
Date: 2026-07-11

## Purpose

Define one consistent semantic model for:

- suffix-BWT tokens;
- virtual document sentinels;
- real byte values;
- alphabet order;
- symbol frequencies;
- `C`;
- `rank`;
- LF mapping.

This obligation prevents an implementation from treating a real `0x00`
byte as the same symbol as a virtual sentinel.

## Dependencies

P5 depends on:

- `GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1`
- `GLYPH_SUFFIX_ARRAY_VALIDITY_V1`
- `GLYPH_SUFFIX_BWT_RELATION_V1`
- `GLYPH_CORPUS_IDENTITY_V1`

## Token universe

A BWT token is exactly one of:

    VIRTUAL_SENTINEL(doc_id)
    BYTE(value)

where:

    0 <= doc_id < document_count
    0 <= value <= 255

Virtual sentinels are not bytes.

They must never be encoded semantically as:

    BYTE(0x00)
    BYTE(0xFF)
    byte + 1
    wrapped uint8 value

## Total order

The canonical token order is:

    VIRTUAL_SENTINEL(0)
    <
    VIRTUAL_SENTINEL(1)
    <
    ...
    <
    VIRTUAL_SENTINEL(document_count - 1)
    <
    BYTE(0x00)
    <
    BYTE(0x01)
    <
    ...
    <
    BYTE(0xFF)

## BWT token sequence

For a suffix-array row beginning at coordinate:

    (doc_id, doc_offset)

the BWT predecessor token is:

    if doc_offset > 0:
        BYTE(document[doc_id][doc_offset - 1])

    if doc_offset == 0:
        VIRTUAL_SENTINEL(doc_id)

The predecessor never crosses into another document.

## Frequency

For token `s`:

    freq(s) = count of positions i such that BWT[i] == s

Every BWT position contributes to exactly one token frequency.

Therefore:

    sum_s freq(s) == len(BWT)

## C array

For token `s`:

    C(s) = sum freq(t) for all t < s

Equivalently:

    C(s) = number of BWT tokens strictly less than s

Required properties:

- `C` is monotonic under canonical token order;
- the first token has `C = 0`;
- adjacent symbols satisfy:

      C(next) = C(current) + freq(current)

- the terminal cumulative value equals BWT length.

## Rank

The canonical rank operation is half-open:

    rank(s, i) = count of s in BWT[0:i]

where:

    0 <= i <= len(BWT)

Therefore:

    rank(s, 0) = 0
    rank(s, len(BWT)) = freq(s)

For every `i < len(BWT)`:

    rank(s, i + 1)
    =
    rank(s, i) + 1, if BWT[i] == s
    rank(s, i),     otherwise

## LF mapping

For BWT row `i`:

    s = BWT[i]

then:

    LF(i) = C(s) + rank(s, i)

The rank argument is `i`, not `i + 1`.

This maps the occurrence at BWT row `i` to the corresponding occurrence
of the same token in the first column.

## LF range

For every BWT row:

    0 <= LF(i) < len(BWT)

LF must be a permutation of:

    0 .. len(BWT) - 1

No two BWT rows may map to the same LF target.

## First-column consistency

Let:

    F = stable_sort(BWT, canonical token order)

Then for every row `i`:

    F[LF(i)] == BWT[i]

The occurrence ordinal must also agree:

    rank(BWT[i], i)

is the zero-based occurrence number of that token in BWT order.

## Sentinel isolation

For every document `d`:

    freq(VIRTUAL_SENTINEL(d)) == 1

A virtual sentinel occurrence must not increment:

    rank(BYTE(0x00), i)

A real zero byte occurrence must not increment:

    rank(VIRTUAL_SENTINEL(d), i)

## Real zero byte

`BYTE(0x00)` is a normal searchable byte token.

It has:

- its own frequency;
- its own `C` entry;
- its own rank stream;
- its own LF transitions.

It is never the end marker.

## Real 0xFF byte

`BYTE(0xFF)` is the largest byte token.

It must not overflow or alias through internal remapping.

Its ordering remains:

    BYTE(0xFE) < BYTE(0xFF)

## Multiple virtual sentinels

Every document has a distinct virtual sentinel.

These are not interchangeable.

Required:

    VIRTUAL_SENTINEL(d1) != VIRTUAL_SENTINEL(d2)

for:

    d1 != d2

Their canonical ordering is by `doc_id`.

## Empty documents

An empty document has no suffix row in the P2 suffix model.

Therefore its virtual sentinel does not appear in the suffix-BWT token
sequence.

P5 frequencies and LF mapping apply only to tokens actually present in BWT.

This does not remove the empty document from P4 corpus identity.

## Required fixtures

The independent oracle must cover:

- empty corpus;
- one non-empty document;
- multiple documents;
- real `0x00`;
- real `0xFF`;
- alternating `0x00` and `0xFF`;
- repeated bytes;
- periodic bytes;
- equal documents;
- prefix-related documents;
- document-start suffixes;
- every byte value `0x00..0xFF`.

## Required mutation failures

The checker must detect:

1. `BYTE(0x00)` replaced by a virtual sentinel;
2. virtual sentinel replaced by `BYTE(0x00)`;
3. wrong sentinel `doc_id`;
4. swapped token order;
5. incorrect frequency;
6. incorrect `C`;
7. inclusive-rank error;
8. exclusive-rank off-by-one error;
9. incorrect LF formula;
10. non-permutation LF;
11. cross-document predecessor token;
12. `0xFF` alias or overflow.

## P5 invariant

For a valid P1–P4 corpus and suffix-BWT sequence:

    token identity
    + canonical order
    + frequency
    + C
    + half-open rank
    + LF

must all describe one identical token sequence.

No layer may use a different interpretation of virtual sentinels or bytes.

## Non-claims

P5 does not yet prove:

- backward-search interval correctness;
- query count correctness;
- locate correctness;
- document-boundary filtering;
- evidence artifact correctness;
- replay correctness.

Those are later proof obligations.

## Completion condition

P5 is complete only when:

1. this specification exists;
2. an independent executable oracle exists;
3. real `0x00` and virtual sentinels remain distinct;
4. `C` matches canonical frequencies;
5. rank is verified at every position;
6. LF is a valid permutation;
7. first-column consistency holds;
8. all mutation fixtures fail;
9. the existing top-level verifier remains green.

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
