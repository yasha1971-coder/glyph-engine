# GLYPH_SITE_PAGE_V1

Status: local site artifact  
Date: 2026-06-28

## Purpose

Record the first professional GLYPH public site page.

The page is designed as a 24-hour reviewer portal, not a generic marketing landing page.

Primary goal:

    allow an external technical reviewer to understand,
    inspect,
    run,
    and challenge the current GLYPH checkpoint without a meeting.

## File

    site/index.html

## Page structure

The page presents:

- GLYPH as replayable exact-byte evidence for fixed corpora
- one-command review checkpoint
- evidence chain
- current verified checkpoint
- binary-safe status
- comparison against search / SIEM / YARA / timestamp layers
- explicit boundaries and non-claims
- reviewer path
- artifact portfolio

## Current strongest claim shown on page

    GLYPH has a reproducible review checkpoint for replayable exact-byte bounded evidence over fixed sentinel-safe corpora.

## Explicit boundaries shown on page

The page explicitly says GLYPH is not currently claiming:

- production binary-safe runtime
- legal proof
- signed/notarized evidence
- distributed search
- SIEM replacement
- Splunk/Elastic replacement
- YARA replacement

## Reviewer flow

The intended reviewer flow is:

    website
    -> GitHub
    -> docs/review/GLYPH_CURRENT_TECHNICAL_STATE_V1.md
    -> docs/review/RLBWT_BOUNDED_EVIDENCE_REVIEW_PATH_V1.md
    -> ./verify.sh

## Status

This is a site artifact stored in the main engineering repository.

Deployment to glyph.rs web root is a separate step.
