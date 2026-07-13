# GLYPH_OPERATOR_QUERY_V1

Status: executable implementation gate
Version: 1
Date: 2026-07-13
Operator obligation: O3

## Purpose

Define exact binary query execution over an O1 committed corpus and its O2
compiled runtime indexes.

O3 is the first operator-facing retrieval action.

It maps:

    binary query
    ->
    compiled C++ FM count and locate
    ->
    canonical (doc_id, doc_offset)
    ->
    committed filesystem path identity

## Dependencies

O3 requires successful verification of:

- O1 — GLYPH_OPERATOR_CORPUS_MANIFEST_V1;
- O2 — GLYPH_OPERATOR_RUNTIME_INDEX_MANIFEST_V1.

The original source directory is not used.

## Query transport

Exactly one authoritative input must be supplied:

    --query-file <regular-file>
    --query-hex <canonical-lowercase-hex>

`--query-file` is preferred for arbitrary binary queries.

The query may contain every byte:

    0x00 through 0xFF

The query must be non-empty.

Shell text, display strings, Unicode decoding, and C-string semantics are not
authoritative query transports.

## Stable query-file requirement

A query file must be:

- a regular file;
- not a symbolic link;
- stable across two complete reads;
- identical in device, inode, size, and modification time;
- byte-identical across both reads.

A changing query file causes failure.

## Runtime authority

Every indexed document is queried through both compiled binaries:

    query_fm_binary_v1
    query_fm_locate_binary_v1

The count and locate results must agree on:

- query identity;
- FM interval;
- match count;
- alphabet size;
- logical sentinel.

The operator layer must not calculate a replacement FM result.

## Query binary commitments

The result binds the exact query binaries used:

- binary name;
- byte size;
- SHA256.

The commitments are checked before and after the complete query.

## Source stability during query

For every committed source snapshot:

1. regular-file identity and SHA256 are checked;
2. count and locate are executed;
3. regular-file identity and SHA256 are checked again;
4. every returned offset is independently byte-checked;
5. final O1 and O2 verification is repeated.

A changed committed snapshot or runtime index invalidates the query.

## Multi-document aggregation

Documents are queried in canonical `doc_id` order.

Global coordinates are ordered by:

    (doc_id, doc_offset)

Physical document concatenation is forbidden.

A byte sequence spanning the end of one document and the start of another is
not a match.

## Bounded locate

The optional limit is:

    --max-offsets N

It applies to the complete ordered multi-document result.

Count remains complete even when locate is bounded.

The result fields are:

- `match_count` — complete number of matches;
- `returned_count` — number of returned coordinates;
- `bounded`;
- `offsets_complete`;
- `max_offsets`.

A limit of zero is valid.

## Source mapping

Every returned result contains:

- `doc_id`;
- `doc_offset`;
- numeric coordinate pair;
- authoritative `relative_path_bytes_hex`;
- non-authoritative `display_path`;
- committed source SHA256;
- successful byte-check flag.

Display text never participates in identity or replay.

## Query result identity

The deterministic `query_result_id` binds:

- result version;
- corpus ID;
- source manifest ID;
- runtime index ID;
- source and runtime manifest hashes;
- query identity;
- query binary commitments;
- max-offset policy;
- per-document FM result digest;
- complete match count;
- returned numeric coordinates;
- bounded and completeness flags.

Absolute filesystem paths, wall-clock time, and duration are excluded.

## Output

The result format is:

    GLYPH_OPERATOR_QUERY_RESULT_V1

Normal JSON is emitted to stdout.

No original source-directory path is included.

## Completion criterion

O3 passes only when:

- query-file and query-hex produce identical semantic results;
- all byte values survive query transport;
- count and locate agree;
- global coordinates match an independent byte oracle;
- invalid UTF-8 path bytes map correctly;
- zero-match and bounded results are correct;
- cross-document-only patterns return zero;
- source and query mutations are rejected;
- runtime mutations are rejected;
- equivalent committed corpora produce byte-identical results.
