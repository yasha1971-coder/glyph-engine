# GLYPH_FM_BACKWARD_SEARCH_EXACTNESS_V1

Status: normative draft  
Version: 1  
Proof obligation: P6  
Date: 2026-07-11

## Purpose

Define and verify exact backward-search semantics over a multi-document byte corpus.

P6 proves that an FM interval represents exactly the suffix-array rows whose
document-local suffixes begin with the requested byte pattern.

The result must agree with an independent naive byte oracle.

## Dependencies

P6 depends on:

- `GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1`
- `GLYPH_SUFFIX_ARRAY_VALIDITY_V1`
- `GLYPH_SUFFIX_BWT_RELATION_V1`
- `GLYPH_CORPUS_IDENTITY_V1`
- `GLYPH_FM_TOKEN_RANK_LF_CONSISTENCY_V1`

## Query domain

A query is an exact byte string:

    pattern = bytes

Every byte value is legal:

    0x00 .. 0xFF

Queries do not contain virtual sentinels.

A virtual sentinel is index structure, not query data.

## Empty query

P6 defines the empty query as invalid.

Required result:

    query_error = EMPTY_PATTERN

The empty query must not silently mean:

- every suffix;
- every byte position;
- the full SA interval;
- one match per document;
- success with count zero.

This avoids ambiguity between search APIs and evidence artifacts.

## Backward-search initialization

For a non-empty pattern:

    l = 0
    r = BWT length

Process bytes from right to left.

For byte token `c`:

    l = C(c) + rank(c, l)
    r = C(c) + rank(c, r)

The interval is half-open:

    [l, r)

If:

    l == r

the pattern is absent.

## Match count

The FM match count is:

    r - l

It must equal the number of valid document-local occurrences.

## Document-local occurrence

A pattern occurs at coordinate:

    (doc_id, doc_offset)

if and only if:

    document[doc_id][
        doc_offset : doc_offset + len(pattern)
    ] == pattern

and:

    doc_offset + len(pattern)
    <= document_length(doc_id)

A match may not cross a document boundary.

## Suffix-array interval semantics

Every SA row inside `[l, r)` must begin with the pattern bytes.

Every SA row outside `[l, r)` must not be a matching row.

The matching rows must form one contiguous interval under canonical suffix order.

## Naive oracle

The independent oracle scans each document separately.

For every document:

    for offset in 0 .. len(document) - len(pattern):
        compare exact bytes

The oracle must not concatenate documents before scanning.

## Boundary rule

For documents:

    doc 0 = b"ab"
    doc 1 = b"cd"

the pattern:

    b"bc"

has zero matches.

Even though the concatenation:

    b"abcd"

contains `b"bc"`.

## Real zero byte

The query:

    b"\x00"

searches real zero bytes.

It must never match a virtual sentinel.

## Real 0xFF byte

The query:

    b"\xff"

searches real `0xFF` bytes.

No internal remapping may alias or overflow the symbol.

## Query longer than document

A pattern longer than a document cannot match that document.

It may still match another longer document in the corpus.

## Duplicate documents

Identical documents produce distinct coordinates because `doc_id` differs.

The FM count includes every occurrence in every document.

## Interval verification

For each query, the checker must prove:

1. FM count equals naive count;
2. FM coordinates equal naive coordinates;
3. every row in `[l, r)` has the pattern prefix;
4. no matching SA row lies outside `[l, r)`;
5. interval is contiguous;
6. no result crosses document boundaries.

## Required fixtures

Fixtures must include:

- empty corpus;
- empty documents;
- ASCII text;
- repeated bytes;
- periodic bytes;
- multiple documents;
- duplicate documents;
- prefix-related documents;
- real `0x00`;
- real `0xFF`;
- alternating `0x00` and `0xFF`;
- full byte alphabet;
- document-boundary false match;
- query absent from corpus;
- query longer than every document;
- query occurring at document end;
- query occurring at offset zero.

## Required query families

The checker must test:

- every one-byte query `0x00..0xFF`;
- selected two-byte binary queries;
- selected repeated patterns;
- complete substrings from fixtures;
- absent patterns;
- cross-boundary-only patterns;
- empty query rejection.

## Required mutation failures

The checker must reject or detect:

1. inclusive rank in backward search;
2. querying sentinel as `0x00`;
3. incorrect `C(BYTE(0x00))`;
4. incorrect `C(BYTE(0xFF))`;
5. wrong initial interval;
6. reversed pattern processed forward;
7. inclusive right interval;
8. cross-document naive oracle;
9. omitted duplicate-document match;
10. fabricated SA row inside interval;
11. missing matching SA row;
12. empty query accepted ambiguously.

## P6 invariant

For every valid non-empty byte pattern:

    FM_interval(pattern)
    ==
    contiguous SA rows for document-local byte-prefix matches

and:

    FM_count(pattern)
    ==
    naive_document_local_count(pattern)

## Non-claims

P6 does not yet prove:

- production locate implementation;
- sampled-SA reconstruction;
- evidence artifact correctness;
- replay bundle correctness;
- completeness claims across omitted offsets;
- binary-safe serialization.

Those are later obligations.

## Completion condition

P6 is complete only when:

1. this specification exists;
2. an independent executable oracle exists;
3. all byte values are searched;
4. `0x00` does not alias a sentinel;
5. cross-document matches are absent;
6. FM intervals equal naive results;
7. interval contents and exclusion are verified;
8. mutation fixtures fail;
9. top-level verification remains green.
