# GLYPH_STRUCTURAL_FINGERPRINT_DIRECTION_V1

Status: active research direction  
Date: 2026-07-01

## Decision

Do not continue positioning GLYPH/STRIDE as a codec predictor.

The tested predictors failed or collapsed under audit:

- entropy
- raw byte r/n
- BWT r/n
- entropy profile variance
- random match-distance
- match-distance tail profile
- useful-anchor repeat-distance profile after LOOCV audit

External work also shows that practical codec-selection systems usually rely on compressor-derived features, many training files, or trial/sampling.

## New direction

GLYPH should frame this branch as:

    deterministic structural fingerprinting of byte corpora

not:

    automatic best-codec prediction

## What a structural fingerprint is

A structural fingerprint is a reproducible artifact describing measurable properties of a corpus:

- source hash
- byte entropy
- byte distribution
- NUL/newline/printable profile
- local entropy profile
- anchor repeat-distance profile
- optional BWT run profile when available

## What it is not

It is not a classifier.

It does not claim to predict bzip2/xz/zstd winners.

It does not replace compressor trials.

It is useful because it gives a reproducible, comparable, byte-exact structural signature.

## GLYPH connection

This aligns with GLYPH's main strength:

    replayable exact-byte evidence over committed corpora

Structural fingerprints can become evidence objects:

    corpus -> measured structure -> replayable JSON artifact

This is a better fit than probabilistic codec prediction.
