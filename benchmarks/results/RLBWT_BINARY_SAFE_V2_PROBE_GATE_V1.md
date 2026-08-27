# RLBWT Binary-Safe V2 Probe Gate V1

- Status: measured synthetic conformance gate; research-tier
- Date: 2026-08-27
- Base commit: `81628f843dc8aa633c65594a66d2386bfc3cb5c2`

## Purpose

Verify a portable, deterministic probe for candidate binary-safe RLBWT
run encodings over canonical `GLYPH_BINARY_BWT_V1` input.

The probe parses and validates the binary BWT container independently,
including the 257-symbol alphabet, logical sentinel 256, 16-bit
little-endian symbols, payload length and FNV-1a64 checksum.

## Gate result

The repository checker verifies:

- four positive synthetic fixtures;
- sixteen fail-closed mutations;
- deterministic output across different working directories;
- deterministic output under different input-item order;
- exact equality of the adaptive-escape upper bound on the hostile
  all-256-symbol fixture;
- canonical path-independent JSON output.

The committed machine-readable result is
`RLBWT_BINARY_SAFE_V2_PROBE_GATE_V1.json`.

## Bound under test

For `R` total BWT runs and one sentinel run, selecting the byte whose
BWT run count is minimal gives adaptive-escape overhead:

    delta = R_escape + 1
    delta <= floor((R - 1) / 256) + 1

This gate contains a constructed equality case. That establishes an
executable check of the arithmetic bound, not a complete runtime-size
claim.

## Reproducibility bindings

    probe SHA-256:
      ef86538cb2ecb07d4cb9fff4770ec0a4f763f9b90a8dbce37182510b83cbcec0
    checker SHA-256:
      a48798e08f5488a0e2ce8696c88d14600940093d975b73c399584aab79f5985b
    result SHA-256:
      16ae8ccfec7132d9b6dce2eed68831bae018384a43b26926a5e6ba9056556598

## Explicit non-claims

This gate does not:

- independently prove that supplied symbols are a suffix BWT;
- select or implement an `RLB2` or `RLR2` format;
- implement compressed rank, LF, count, locate or extraction;
- include locate samples, metadata or evidence bytes;
- establish a complete binary-safe runtime below corpus size;
- replace the required `GLYPH_BINARY_RUNTIME_V1`;
- change or validate the live public demo.

The next gate is a portable real-corpus matrix followed by a complete
rank, locate and metadata byte budget.
