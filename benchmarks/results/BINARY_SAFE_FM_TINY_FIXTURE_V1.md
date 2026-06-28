# BINARY_SAFE_FM_TINY_FIXTURE_V1

Status: local binary-safe FM reference fixture  
Date: 2026-06-27

## Purpose

Validate binary-safe exact query/count/locate semantics on a tiny corpus containing `0x00`.

This is a reference Python fixture, not the optimized C++ runtime.

It validates the design decision:

- source bytes may contain `0x00`
- query bytes may contain `0x00`
- `0x00` is preserved as data
- virtual sentinel is symbol `256`
- exact count must mean real non-wrapping source occurrences
- returned offsets must byte-check

## Tool

    tools/run_binary_safe_fm_tiny_fixture_v1.py

## Corpus

Bytes:

    41 00 42 00 43 00 42

Visualization:

    A\0B\0C\0B

Source bytes:

    7

NUL bytes:

    3

## Query

Bytes:

    00 42

Expected result:

- match_count: 2
- offsets: [1, 5]
- byte_check: true

## Symbol model

Data symbols:

    0..255

Virtual sentinel:

    256

Symbol stream:

    65, 0, 66, 0, 67, 0, 66, 256

## Validated result

The reference FM fixture returns:

- exact FM interval
- match_count: 2
- offsets: [1, 5]
- byte_check: true
- no rejected boundary-crossing offsets

## Interpretation

This is the first positive binary-safe query/count/locate fixture for GLYPH semantics.

It does not yet claim that the production GLYPH C++ runtime is binary-safe.

It defines the reference behavior that future C++ binary-safe runtime must match.
