# GLYPH_LOCATE_COORDINATE_EXACTNESS_V1

Status: normative draft  
Version: 1  
Proof obligation: P7  
Date: 2026-07-11

## Purpose

Define exact locate semantics for GLYPH over a multi-document byte corpus.

P7 proves that:

- FM count;
- full locate;
- bounded locate;
- returned coordinates;
- byte verification;
- bounded/completeness metadata

all describe the same document-local match set.

## Dependencies

P7 depends on:

- `GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1`
- `GLYPH_SUFFIX_ARRAY_VALIDITY_V1`
- `GLYPH_SUFFIX_BWT_RELATION_V1`
- `GLYPH_CORPUS_IDENTITY_V1`
- `GLYPH_FM_TOKEN_RANK_LF_CONSISTENCY_V1`
- `GLYPH_FM_BACKWARD_SEARCH_EXACTNESS_V1`

## Canonical coordinate

Every occurrence is represented as:

    (doc_id, doc_offset)

where:

    0 <= doc_id < document_count
    0 <= doc_offset < document_length(doc_id)

A raw concatenated offset is not the canonical P7 coordinate.

## Coordinate meaning

For non-empty pattern `p`, coordinate `(d, o)` is valid if and only if:

    document[d][o:o+len(p)] == p

and:

    o + len(p) <= len(document[d])

The match must remain inside one document.

## Full locate

For FM interval:

    [l, r)

full locate returns exactly the coordinates represented by:

    suffix_array[l:r]

After coordinate conversion, the public canonical result order is:

    ascending (doc_id, doc_offset)

Full locate must contain:

- every valid match;
- no invalid match;
- no duplicates.

## Count consistency

The exact match count is:

    match_count = r - l

Required:

    match_count
    ==
    len(full_locate_coordinates)
    ==
    naive_document_local_count

## Bounded locate

Bounded locate accepts:

    max_offsets >= 0

It returns the canonical prefix of the full locate list:

    offsets =
        full_offsets[0:min(match_count, max_offsets)]

Required:

    returned_count = len(offsets)

    returned_count
    =
    min(match_count, max_offsets)

## Bounded flag

The canonical bounded flag is:

    bounded = returned_count < match_count

Therefore:

- if all matches are returned, `bounded = false`;
- if any matches are omitted, `bounded = true`;
- zero matches always produce `bounded = false`;
- `max_offsets = 0` with positive count produces `bounded = true`.

## Completeness meaning

For full locate:

    offsets_complete = true

For bounded locate:

    offsets_complete = not bounded

A bounded response must never imply that omitted coordinates do not exist.

## Byte check

Every returned coordinate must pass:

    source_bytes_at_coordinate == query_bytes

Byte verification must be length-aware.

It must not use:

- C-string termination;
- `strlen`;
- `strcmp`;
- `strstr`;
- implicit NUL truncation.

## Real zero byte

Queries and matched spans may contain `0x00`.

The byte check must compare the full explicit pattern length.

## Real 0xFF byte

Queries and matched spans may contain `0xFF`.

No sign extension, overflow, or sentinel alias is permitted.

## Duplicate documents

Identical source bytes in different documents produce different coordinates.

Example:

    doc 0 = b"same"
    doc 1 = b"same"

The match at offset zero exists twice:

    (0, 0)
    (1, 0)

## Empty documents

Empty documents produce no valid locate coordinates.

They remain part of corpus identity under P4.

## Empty query

The empty query is invalid:

    query_error = EMPTY_PATTERN

P7 must not return synthetic coordinates for an empty query.

## Canonical ordering

Returned public coordinates are ordered by:

    doc_id ascending
    then doc_offset ascending

This ordering is independent of:

- suffix-array row order;
- thread scheduling;
- hash-map iteration;
- locate traversal order;
- filesystem order.

## Full result requirements

A full locate result must contain:

- `fm_interval`;
- `match_count`;
- `coordinates`;
- `returned_count`;
- `bounded = false`;
- `offsets_complete = true`;
- `byte_check = true`.

## Bounded result requirements

A bounded locate result must contain:

- the same `fm_interval`;
- the same exact `match_count`;
- canonical-prefix coordinates;
- `returned_count`;
- exact `max_offsets`;
- correct `bounded`;
- correct `offsets_complete`;
- `byte_check = true`.

## Count-only relation

A count-only result may omit coordinates, but it must preserve:

    fm_interval
    match_count

Count-only mode must not claim:

    offsets_complete = true

unless full locate was actually performed.

## Required fixtures

P7 fixtures must include:

- empty corpus;
- empty documents;
- one match;
- zero matches;
- many matches;
- repeated bytes;
- periodic bytes;
- duplicate documents;
- prefix-related documents;
- matches at offset zero;
- matches ending at document end;
- real `0x00`;
- real `0xFF`;
- patterns containing both `0x00` and `0xFF`;
- cross-document-only false match;
- full byte alphabet.

## Required bounds

For each positive query, test:

    max_offsets = 0
    max_offsets = 1
    max_offsets = match_count - 1, when possible
    max_offsets = match_count
    max_offsets = match_count + 1

## Required mutation failures

The checker must detect:

1. missing coordinate;
2. fabricated coordinate;
3. duplicate coordinate;
4. wrong `doc_id`;
5. wrong `doc_offset`;
6. cross-document coordinate;
7. coordinate whose bytes do not match;
8. count different from interval width;
9. count different from full locate size;
10. wrong `returned_count`;
11. wrong `bounded` flag;
12. wrong `offsets_complete`;
13. bounded result not equal to canonical prefix;
14. unstable coordinate ordering;
15. NUL-truncated byte check;
16. empty query accepted;
17. negative `max_offsets`;
18. duplicate-document occurrence collapsed.

## P7 invariant

For every valid non-empty byte pattern:

    FM count
    ==
    naive count
    ==
    full locate size

and:

    bounded locate
    ==
    canonical prefix of full locate

Every returned coordinate must independently reproduce the exact query bytes.

## Non-claims

P7 does not yet prove:

- sampled-SA LF reconstruction in a production runtime;
- serialization safety;
- evidence artifact integrity;
- replay bundle integrity;
- tamper detection;
- membership or non-membership proofs.

Those are later proof obligations.

## Completion condition

P7 is complete only when:

1. this specification exists;
2. an independent executable oracle exists;
3. full locate equals naive coordinates;
4. bounded locate is a canonical prefix;
5. count/full/bounded results agree;
6. every returned coordinate passes explicit-length byte check;
7. binary `0x00` and `0xFF` fixtures pass;
8. cross-document results are absent;
9. all mutation fixtures fail;
10. top-level verification remains green.
