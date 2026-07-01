# GLYPH_STRUCTURAL_FINGERPRINT_NON_PREDICTION_V1

Status: active boundary  
Date: 2026-07-01

## Purpose

Define what `GLYPH_STRUCTURAL_FINGERPRINT_V0` is and what it is not.

This document exists to prevent a wrong interpretation of Structural Fingerprint V0 as a codec predictor, compression optimizer, or machine-learning classifier.

## Core statement

`GLYPH_STRUCTURAL_FINGERPRINT_V0` is a deterministic, replayable structural measurement artifact over source bytes.

It is not a prediction system.

It records reproducible facts about a committed byte corpus:

- byte identity
- source SHA256
- byte statistics
- entropy profile
- anchor repeat profile
- optional BWT run profile
- explicit non-claims

A third party can regenerate the artifact from the same source bytes and verify that the structural measurements replay.

## What it is not

Structural Fingerprint V0 is not:

- a codec predictor
- a compression optimizer
- a codec router
- a replacement for compressor trials
- a machine-learning model
- a claim that one codec will beat another
- a statistical generalization over file types
- a benchmark result

## Why this boundary exists

GLYPH/STRIDE tested several codec-prediction hypotheses:

- entropy
- raw run ratio
- BWT run ratio
- entropy profile variance
- match-distance profile
- anchor repeat-distance profile

The stronger-looking signals failed under broader testing or anti-overfit audit.

The final conclusion was:

    codec prediction from small-sample byte statistics is not a reliable GLYPH claim.

Therefore GLYPH should not position Structural Fingerprint V0 as a predictor.

## Correct positioning

The correct positioning is:

    reproducible structural corpus measurement

not:

    automatic codec selection

This extends GLYPH's evidence model:

    exact-byte retrieval evidence
    +
    replayable structural measurement

## Relationship to external codec-prediction work

Existing codec-prediction research generally relies on:

- large training sets
- compressor-derived features
- sampling or trial compression
- machine learning models

GLYPH Structural Fingerprint V0 deliberately does something different:

    It produces deterministic measurements that can be independently replayed.

It does not try to beat ML codec predictors.

## Public wording

Safe wording:

    GLYPH can produce a replayable structural fingerprint of a byte corpus.

Unsafe wording:

    GLYPH predicts the best compressor.
    GLYPH chooses the optimal codec.
    GLYPH optimizes compression.
    GLYPH classifies file types.

## Verification path

Current replay path:

    python3 tools/glyph_structural_fingerprint_v0.py <source> --out <artifact>
    python3 tools/replay_structural_fingerprint_v0.py <artifact> --out <replay>

Top-level smoke test:

    ./verify.sh

Expected lines:

    [verify] Structural Fingerprint V0 replay smoke
    [verify] structural fingerprint replay ok
    VERIFY OK

## Decision

Keep Structural Fingerprint V0 as a replayable evidence artifact.

Do not use it as a codec-prediction claim unless a future separate study proves out-of-sample predictive value on a large external dataset.
