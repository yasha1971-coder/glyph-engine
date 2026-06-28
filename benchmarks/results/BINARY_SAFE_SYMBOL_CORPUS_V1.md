# BINARY_SAFE_SYMBOL_CORPUS_V1

Status: local symbol-corpus validation  
Date: 2026-06-27

## Purpose

Validate the first binary-safe construction layer for GLYPH.

This does not build SA/BWT/FM yet.

It validates the symbol model selected in:

- `docs/specs/BINARY_SAFE_DESIGN_DECISION_V1.md`

## Tooling

Builder:

    tools/make_binary_safe_symbol_corpus_v1.py

Verifier:

    tools/verify_binary_safe_symbol_corpus_v1.py

## Input corpus

Bytes:

    41 00 42 00 43 00 42

Visualization:

    A\0B\0C\0B

NUL bytes:

    3

## Symbol model

Data bytes:

    0..255

Virtual sentinel:

    256

Encoding:

    u16le

Expected symbol stream:

    65, 0, 66, 0, 67, 0, 66, 256

## Validated invariants

- symbol_count equals source_bytes + 1
- exactly one virtual sentinel
- 0x00 is preserved as data symbol 0
- sentinel is last symbol
- all source symbols are in 0..255
- sentinel is not a corpus byte

## Result

Binary-safe symbol corpus V1 passed local validation.

This is the first construction layer toward binary-safe GLYPH.

The current full GLYPH v0.x builder remains sentinel-safe; full binary-safe index/runtime/evidence support is not claimed yet.
