# Sentinel invariant

GLYPH FM-index v0.x is built over:

    raw_corpus + real appended 0x00 sentinel

The sentinel is a real byte appended to the indexed corpus before suffix array, BWT, and FM-index construction.

This invariant is required for byte-exact FM correctness.

## Current v0.x constraint

Input corpora must not contain `0x00`.

The `0x00` byte is reserved as the unique terminal sentinel for the indexed corpus.

## Why this matters

The suffix array, BWT, and FM-index must describe the same byte sequence.

Using a synthetic sentinel during BWT construction without appending it to the corpus can create inconsistent FM intervals and undercount matches.

## Canonical builder

Use:

    tools/build_glyph_index_v1.sh

Do not bypass the canonical builder for v0.x indexes.

## Future direction

A future index format may support arbitrary `0x00` bytes in input corpora by using a 257-symbol alphabet or an explicit out-of-band sentinel representation.
