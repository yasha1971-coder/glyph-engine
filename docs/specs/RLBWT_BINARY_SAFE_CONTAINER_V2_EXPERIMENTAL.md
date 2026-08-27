# RLBWT Binary-Safe Container V2 — Experimental

Status: experimental additive format
Version: 2

## Purpose

RLB2 stores the canonical binary-safe BWT as runs while preserving alphabet
257 and logical sentinel 256. It is an experimental research container and
does not redefine RLB1 or RLR1.

## Identity

- magic: `GLYRLB2\0`
- version: 2
- fixed header size: 160 bytes
- alphabet size: 257
- logical sentinel: 256
- decoded target: canonical uint16 little-endian BWT payload

The header binds source-BWT identity, decoded dimensions and encoded payload
identity. Implementations validate declared lengths, checksums and digests
before accepting an artifact.

## Run-head encoding

One escape byte is selected deterministically from source-byte symbols by
minimum BWT-run count with deterministic numeric tie-breaking.

- ordinary source symbol: one byte;
- selected escape source symbol: `escape`, tag 0;
- logical sentinel 256: `escape`, tag 1.

Other escape tags are invalid.

## Run-length encoding

Every run length is a positive canonical ULEB128 integer.

Zero, overflow, truncation and non-canonical ULEB128 representations are
rejected.

## Required validation

A conforming reader rejects:

- wrong magic, version, header size or constants;
- malformed or non-canonical integer encodings;
- invalid escape tags;
- zero run lengths;
- row-count or run-count disagreement;
- truncated or trailing payload data;
- payload checksum or SHA-256 disagreement;
- decoded source checksum or SHA-256 disagreement;
- output paths that already exist.

Encode and decode publication must be fail-closed and must not overwrite an
existing output.

## Scope boundary

RLB2 is currently a compressed BWT container with encode, inspect and decode
operations. It is not a rank index, locate index, query server, full GLYPH
runtime or general-purpose replacement for a lossless codec.

Any complete-runtime size claim must include RLB2, rank, locate, metadata and
all artifacts required by the claimed query and verification path.
