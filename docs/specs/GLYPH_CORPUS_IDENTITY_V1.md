# GLYPH_CORPUS_IDENTITY_V1

Status: normative draft  
Version: 1  
Proof obligation: P4  
Date: 2026-07-11

## Purpose

Define the canonical identity of a GLYPH corpus.

This specification fixes:

- format version;
- ordered document manifest;
- source-byte identity;
- canonical document coordinates;
- corpus-ID preimage;
- corpus-ID digest;
- empty-document semantics;
- document-order semantics.

Without P4, two implementations may reproduce the same local measurements while referring to different logical corpora.

## Dependencies

P4 is the identity foundation used by later proof obligations.

It is compatible with:

- `GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1`
- `GLYPH_SUFFIX_ARRAY_VALIDITY_V1`
- `GLYPH_SUFFIX_BWT_RELATION_V1`

## Core rule

A GLYPH corpus is not identified only by concatenated bytes.

A canonical corpus identity includes:

    format version
    ordered document manifest
    document identifiers
    document lengths
    document SHA256 values
    exact source bytes

Changing any identity-bearing field changes `corpus_id`.

## Corpus model

A logical corpus is an ordered sequence:

    C = [D_0, D_1, ..., D_(k-1)]

Each document contains:

- canonical `doc_id`;
- canonical `doc_name`;
- exact byte content;
- byte length;
- SHA256 digest.

Documents may be empty.

Documents may contain every byte value:

    0x00 .. 0xFF

## Canonical document ID

Document IDs are consecutive unsigned integers:

    0, 1, 2, ..., document_count - 1

The document ID is defined by manifest order.

It is not derived from:

- filesystem inode;
- directory traversal order;
- locale sorting;
- modification time;
- absolute path;
- process scheduling.

## Canonical document name

Each document has a UTF-8 name.

The name is identity-bearing.

Names must satisfy:

- valid Unicode scalar sequence;
- encoded as UTF-8;
- no Unicode normalization performed implicitly;
- no NUL character;
- unique within the manifest;
- preserved exactly as supplied.

The byte sequence of the UTF-8 encoding is part of the identity preimage.

## Canonical coordinate

A corpus byte coordinate is:

    (doc_id, doc_offset)

with:

    0 <= doc_id < document_count
    0 <= doc_offset < document_length(doc_id)

Coordinates are never represented normatively as one global concatenated offset.

A global physical offset may exist as an implementation detail, but evidence artifacts must preserve:

    doc_id
    doc_offset

## Empty documents

An empty document:

- remains present in the ordered manifest;
- has byte length zero;
- has SHA256 of the empty byte string;
- contributes no valid byte coordinates;
- contributes no suffix coordinates;
- still affects corpus identity.

Removing, adding, or reordering an empty document changes `corpus_id`.

## Ordered manifest

The ordered manifest is identity-bearing.

These corpora are different:

    [A, B]
    [B, A]

even when concatenated bytes happen to be equal.

Document boundaries are identity-bearing.

These corpora are different:

    ["ab", "cd"]
    ["abc", "d"]

even though both concatenate to:

    "abcd"

## Format identity

The normative format identifier is:

    GLYPH_CORPUS_IDENTITY_V1

The format identifier is included in the corpus-ID preimage.

A future version must use a different identifier.

Changing semantic rules without changing the format identifier is forbidden.

## Hash function

P4 uses SHA-256.

Digest encoding is lowercase hexadecimal when serialized for display.

The binary digest is 32 bytes.

## Document digest

For document `D_d`:

    document_sha256[d] = SHA256(document_bytes)

The document digest is calculated over the exact bytes only.

It does not include:

- name;
- document ID;
- length;
- path;
- timestamps;
- metadata.

Those values enter the higher-level corpus preimage separately.

## Canonical integer encoding

All integers in the corpus-ID preimage use:

    unsigned 64-bit big-endian

This includes:

- document count;
- document ID;
- document-name byte length;
- document byte length.

Values outside:

    0 .. 2^64 - 1

are invalid.

## Domain-separated corpus-ID preimage

The corpus-ID preimage is a binary sequence.

It begins with:

    ASCII("GLYPH_CORPUS_IDENTITY_V1")
    BYTE(0x00)

Then:

    U64_BE(document_count)

For every document in manifest order:

    U64_BE(doc_id)
    U64_BE(name_utf8_length)
    name_utf8_bytes
    U64_BE(document_byte_length)
    document_sha256_binary

No delimiter ambiguity exists because all variable-length fields are length-prefixed.

## Corpus ID

The canonical corpus ID is:

    corpus_id = SHA256(corpus_id_preimage)

Display form:

    lowercase hexadecimal, 64 characters

## Manifest serialization

Canonical JSON is allowed as an artifact representation, but JSON text is not itself the corpus-ID preimage.

The normative identity is computed from the binary preimage defined above.

JSON serializers may vary in whitespace without changing `corpus_id`.

## Canonical manifest fields

A P4 manifest contains:

    format
    document_count
    total_bytes
    documents
    corpus_id_preimage_sha256
    corpus_id

Each document entry contains:

    doc_id
    doc_name
    byte_length
    sha256

## Preimage self-check

For V1:

    corpus_id_preimage_sha256 == corpus_id

Both fields may be emitted for explicitness, but they are the same SHA-256 digest over the normative preimage.

Future formats may separate identity layers, but V1 does not.

## Total byte count

The manifest records:

    total_bytes = sum(document byte lengths)

`total_bytes` is derived metadata.

It must agree with the document entries.

It is not separately appended to the identity preimage because it is already determined by the ordered document lengths.

## Path handling

Filesystem paths are not identity-bearing unless explicitly supplied as `doc_name`.

Absolute local paths must not enter the canonical corpus ID accidentally.

The same corpus copied to another directory must retain the same corpus identity when:

- ordered document names are unchanged;
- document bytes are unchanged;
- format version is unchanged.

## Mutation rules

Each of these must change corpus identity:

- one-byte source mutation;
- document insertion;
- document deletion;
- document reordering;
- document renaming;
- boundary repartitioning;
- empty-document insertion;
- empty-document removal;
- format-version change;
- real `0x00` mutation;
- real `0xFF` mutation.

## Non-identity metadata

These values must not change corpus identity unless promoted into a future format:

- local absolute path;
- file modification time;
- file permissions;
- filesystem inode;
- owner;
- compression container;
- host name;
- build machine;
- thread count.

## Required fixtures

Positive fixtures must include:

- empty corpus;
- one empty document;
- multiple empty documents;
- one-byte `0x00`;
- one-byte `0xFF`;
- full byte alphabet;
- duplicate byte content with different names;
- duplicate document names rejected;
- same bytes at different paths;
- multiple documents;
- boundary-sensitive equal-concatenation corpus pair;
- reordered corpus pair.

## Required rejection classes

A P4 validator must reject:

1. non-consecutive document IDs;
2. duplicate document names;
3. invalid UTF-8 names;
4. NUL inside document name;
5. incorrect document length;
6. incorrect document SHA256;
7. incorrect total byte count;
8. incorrect document count;
9. incorrect format identifier;
10. incorrect corpus ID;
11. incorrect manifest order;
12. malformed SHA256 encoding.

## Boundary example

These corpora have equal concatenated bytes:

    Corpus A:
        doc 0 = "ab"
        doc 1 = "cd"

    Corpus B:
        doc 0 = "abc"
        doc 1 = "d"

They must have different corpus IDs because their document manifests differ.

## Reordering example

These corpora contain the same document set:

    Corpus A = [alpha, beta]
    Corpus B = [beta, alpha]

They must have different corpus IDs.

## Duplicate-content example

Two documents may contain identical bytes:

    doc 0 name = "copy-a"
    doc 1 name = "copy-b"

They remain distinct documents because their:

- document IDs;
- names;
- manifest positions

are different.

## Determinism

For identical:

- format identifier;
- ordered document names;
- ordered document bytes;

all conforming implementations must produce the same:

- document digests;
- identity preimage;
- corpus ID;
- canonical coordinates.

The result must not depend on local paths, timestamps, or host architecture.

## P4 invariant

Two P4 corpora have the same `corpus_id` if and only if their canonical V1 identity preimages are byte-identical, assuming SHA-256 collision resistance.

## Non-claims

P4 does not prove:

- suffix-array correctness;
- BWT correctness;
- FM-index correctness;
- query correctness;
- document-boundary filtering;
- locate correctness;
- evidence-bundle completeness;
- cryptographic provenance of the original acquisition process.

It proves only canonical corpus identity under the V1 model.

## Completion condition

P4 is complete only when:

1. this specification exists;
2. an executable independent identity oracle exists;
3. positive fixtures reproduce deterministically;
4. equal concatenation with different boundaries yields different IDs;
5. document reordering yields a different ID;
6. empty-document changes yield a different ID;
7. byte mutations yield a different ID;
8. malformed manifests are rejected;
9. top-level verification remains green.
