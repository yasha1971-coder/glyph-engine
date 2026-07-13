# GLYPH_BINARY_SAFE_QUERY_TRANSPORT_V1

Status: normative draft  
Version: 1  
Proof obligation: P8  
Date: 2026-07-11

## Purpose

Define binary-safe transport, serialization, decoding, hashing, and replay rules
for GLYPH queries and matched byte spans.

P8 prevents silent truncation or reinterpretation of queries containing:

- `0x00`;
- `0xFF`;
- non-UTF-8 byte sequences;
- embedded control bytes;
- repeated zero bytes.

## Dependencies

P8 depends on P1 through P7.

## Canonical query representation

The authoritative serialized query representation is:

    query_hex

Rules:

- lowercase hexadecimal;
- exactly two hex characters per byte;
- even length;
- no prefix such as `0x`;
- no whitespace;
- no separators;
- no Unicode normalization;
- no implicit text decoding.

Examples:

    bytes: 00
    query_hex: "00"

    bytes: 00 ff 41
    query_hex: "00ff41"

## Query length

Artifacts must include:

    query_length_bytes

Required invariant:

    query_length_bytes == len(decode_hex(query_hex))

An empty query is invalid:

    query_length_bytes > 0

## Query hash

Artifacts must include:

    query_sha256

The preimage is exactly the decoded query byte sequence:

    SHA256(query_bytes)

It is not the hash of:

- the hexadecimal text;
- a UTF-8 representation;
- a shell argument;
- a JSON string with escapes;
- a NUL-terminated prefix.

## Display text

A human-readable query field may exist only as non-authoritative metadata.

It must be labelled:

    query_display

It must never be used for:

- search;
- hashing;
- replay;
- byte checking;
- artifact identity.

For arbitrary binary input, `query_display` may be absent.

## JSON transport

JSON carries the query as `query_hex`.

JSON must not transport arbitrary query bytes as a raw text string.

JSON escaping is not a binary transport format.

## Command-line transport

Binary queries must not be passed through shell variables as raw bytes.

Canonical supported paths are:

    --query-hex <lowercase-even-hex>

or:

    --query-file <binary-file>

A query file is read with explicit byte length.

## Standard input

If standard input is supported, it must be read as bytes until EOF.

It must not use line-oriented text APIs.

## Decoding

The decoder must reject:

- odd-length hex;
- uppercase hex;
- whitespace;
- non-hex characters;
- `0x` prefixes;
- empty hex;
- mismatched declared length;
- mismatched SHA256.

No repair or normalization is allowed.

## Byte checking

For coordinate `(doc_id, doc_offset)` and query length `m`:

    candidate =
        document[doc_id][doc_offset:doc_offset+m]

Required:

    len(candidate) == m
    candidate == query_bytes

The comparison must use explicit length.

Forbidden implementations include:

- `strlen`;
- `strcmp`;
- `strncmp` with derived C-string length;
- `strstr`;
- `%s`;
- APIs that stop at `0x00`.

## Replay

Replay must reconstruct query bytes only from canonical `query_hex`.

Replay then verifies:

1. canonical hex encoding;
2. query length;
3. query SHA256;
4. FM interval metadata, where present;
5. match count metadata, where present;
6. every returned coordinate;
7. exact byte span at every coordinate.

A replay verifier must reject mutation of any authoritative query field.

## Artifact consistency

The following fields are mutually bound:

    query_hex
    query_length_bytes
    query_sha256

Changing any one without changing the others must fail replay.

## Binary fixtures

Required query fixtures include:

- `00`;
- `ff`;
- `00ff`;
- `ff00`;
- `0000`;
- `ffff`;
- `410042`;
- `00ff0041ff`;
- invalid UTF-8 sequences;
- all bytes `00..ff`;
- byte sequences containing newline and carriage return.

## Mutation requirements

The checker must reject:

1. uppercase hex;
2. odd-length hex;
3. whitespace in hex;
4. `0x` prefix;
5. invalid hex digit;
6. empty query;
7. wrong query length;
8. wrong query SHA256;
9. truncated query at first NUL;
10. query hash computed from hex text;
11. changed coordinate;
12. changed source byte;
13. false `byte_check=true`;
14. use of display text as replay source;
15. extra trailing byte omitted from comparison;
16. removal of a zero byte;
17. replacement of `0xFF`;
18. JSON round trip that changes bytes.

## P8 invariant

For every valid artifact:

    decoded_query
    ==
    original_query_bytes

and:

    query_length_bytes
    ==
    len(decoded_query)

and:

    query_sha256
    ==
    SHA256(decoded_query)

and every returned coordinate reproduces the complete decoded query byte sequence.

## Non-claims

P8 does not yet prove:

- evidence bundle integrity;
- artifact schema completeness;
- tamper-evident manifests;
- portable path handling;
- membership or non-membership proof semantics.

Those are later obligations.

## Completion condition

P8 is complete only when:

1. this specification exists;
2. a canonical encoder and strict decoder exist;
3. all 256 byte values survive transport;
4. embedded NUL bytes survive transport;
5. query length and hash are verified;
6. byte checking uses explicit length;
7. mutation fixtures fail;
8. JSON round trips preserve exact bytes;
9. top-level verification remains green.

## Coherent query replacement boundary

P8 detects corruption when any authoritative query-binding field disagrees:

    decoded_length(query_hex) != query_length_bytes

or:

    SHA256(decoded_query_bytes) != query_sha256

A mutation that consistently replaces all of the following:

- `query_hex`;
- `query_length_bytes`;
- `query_sha256`;
- coordinates and match count when necessary;

is not transport corruption. It is a newly constructed artifact for a different
query.

P8 cannot infer the prior intended query from a fully self-consistent
replacement. Binding the complete authoritative artifact against replacement
belongs to P10 replay determinism and artifact identity.

Therefore the `trailing_byte_omitted` mutation truncates `query_hex` while
retaining the original declared byte length and SHA256.
