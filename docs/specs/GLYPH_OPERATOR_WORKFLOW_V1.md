# GLYPH_OPERATOR_WORKFLOW_V1

Status: executable implementation gate
Version: 1
Date: 2026-07-14
Operator obligation: O5

## Purpose

Define the first complete one-command GLYPH operator workflow.

The operator supplies:

- an ordinary filesystem directory;
- one exact binary query;
- an output directory that does not yet exist;
- an optional global locate limit.

The workflow performs O1 through O4 without requiring the operator to invoke
the internal stages separately.

## Command

The canonical command is:

    ./tools/glyph_operator_workflow_v1.py run \
        --source <ordinary-directory> \
        --query-file <binary-query-file> \
        --out <new-case-directory>

Canonical hexadecimal transport is also supported:

    ./tools/glyph_operator_workflow_v1.py run \
        --source <ordinary-directory> \
        --query-hex <lowercase-hex> \
        --out <new-case-directory>

The optional global locate bound is:

    --max-offsets N

## Workflow

The command performs:

    O1 deterministic source snapshot
    ->
    O2 compiled binary-safe runtime indexes
    ->
    O3 exact compiled count and locate
    ->
    O4 self-contained evidence bundle
    ->
    O5 case-level closure and atomic publication

## Output layout

The final case directory contains exactly:

    corpus/
    bundle/
    workflow_result_v1.json
    WORKFLOW_COMPLETE_V1.json

`corpus/` contains the O1 and O2 reusable committed corpus.

`bundle/` contains the independently replayable O4 evidence bundle.

## Input authority

The original source directory is used only by O1 while constructing the
committed snapshot.

O2, O3, O4, final verification, and later replay use committed files.

The original source directory is not part of any result identity.

## Query transport

Exactly one query input is required:

    --query-file
    --query-hex

The query must be non-empty.

Query files must satisfy the O3 stable regular-file requirements.

All bytes from `0x00` through `0xFF` are valid query bytes.

## Atomic publication

The complete case is constructed in a temporary sibling directory.

The requested output path must not already exist.

The temporary case must pass:

- O1 verification;
- O2 verification;
- O3 artifact comparison;
- O4 standalone bundle replay;
- O5 workflow verification.

Only then is it renamed to the requested final path.

A failed or interrupted workflow must not publish the final output path.

## Path safety

The output must not be:

- equal to the source directory;
- inside the source directory;
- an ancestor of the source directory;
- an existing path.

This prevents recursive corpus capture and accidental replacement.

## Workflow result

`workflow_result_v1.json` binds:

- corpus ID;
- source manifest ID;
- runtime index ID;
- query result ID;
- bundle root SHA256;
- bundle manifest SHA256;
- exact query identity;
- global locate policy;
- document and match counts;
- fixed output layout.

Absolute paths, timestamps, duration, host identity, and temporary paths are
excluded.

## Complete marker

`WORKFLOW_COMPLETE_V1.json` binds:

- workflow-result SHA256;
- workflow-result ID;
- corpus ID;
- runtime index ID;
- query result ID;
- bundle root SHA256.

## Verification command

A completed case is checked with:

    ./tools/glyph_operator_workflow_v1.py verify \
        --case <case-directory>

Verification recomputes all identities and invokes the O1, O2, and O4
verifiers.

## Portability boundary

The complete case is intended for continued work in a GLYPH checkout.

The contained `bundle/` remains independently replayable without:

- the original source directory;
- the GLYPH repository;
- network access;
- external data.

## Completion criterion

O5 passes only when:

- one command performs O1 through O4;
- query-file and query-hex paths are supported;
- equivalent inputs produce byte-identical case trees;
- the case verifies after the original source is removed;
- the contained bundle replays outside the repository;
- final output publication is atomic;
- interrupted execution leaves no final output;
- malformed and mutated cases are rejected.
