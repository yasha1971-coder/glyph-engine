# BINARY_SAFE_ROADMAP_V1

Status: draft  
Date: 2026-06-27

## Purpose

Define the transition from the current sentinel-safe GLYPH v0.x model to a binary-safe exact-byte retrieval model.

Current v0.x invariant:

    source corpus must not contain 0x00
    index corpus = source corpus + physical 0x00 sentinel

This is acceptable for text/log/legal/source corpora that contain no NUL bytes.

It is not acceptable for arbitrary binary corpora such as:

- PE/ELF binaries
- malware samples
- memory dumps
- packet captures
- raw forensic images
- mixed binary logs

## Target invariant

Binary-safe GLYPH must treat every byte value as data:

    0x00..0xFF are valid corpus bytes

No corpus byte may be reserved as an in-band terminator.

The engine must be length-driven, not NUL-terminated.

## Required properties

Binary-safe GLYPH must support:

- exact query bytes containing 0x00
- source corpora containing 0x00
- byte offsets unaffected by encoding, line endings, Unicode, or C-string termination
- byte_check over returned offsets
- replay verification
- portable evidence bundle
- schema validation

## Sentinel model transition

The current physical sentinel model:

    raw_corpus + 0x00

must be replaced by one of:

1. virtual sentinel symbol outside byte alphabet

       data alphabet: 0..255
       virtual sentinel: 256

2. coordinate boundary model

       data alphabet: 0..255
       boundary stored as metadata / primary_index

For GLYPH evidence semantics, the selected model must prevent false boundary-crossing matches.

Exact match_count must not include matches that only exist because of cyclic wrap-around.

## Hot-loop requirement

C++ hot loops must not rely on:

- strlen
- C-string termination
- byte == 0 as end-of-data
- sentinel byte inside user data

All loops must be bounded by explicit lengths.

## Current boundary test

A minimal binary corpus:

    41 00 42 00 43 00 42

Textual visualization:

    A\0B\0C\0B

Query:

    00 42

Expected future binary-safe result:

    match_count: 2
    offsets: [1, 5]
    byte_check: true

Under current sentinel-safe v0.x, this must fail at corpus preparation or indexing.

That failure is expected and should remain documented until binary-safe runtime support is implemented.

## Non-claims

This roadmap does not claim that current GLYPH is binary-safe.

This roadmap defines the boundary and the target migration path.
