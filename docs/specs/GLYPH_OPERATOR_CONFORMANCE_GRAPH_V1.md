# GLYPH_OPERATOR_CONFORMANCE_GRAPH_V1

Status: executable conformance closure
Version: 1
Date: 2026-07-14
Operator obligation: O6

## Purpose

Close the executable GLYPH operator path.

O6 proves that the verified reference semantics and compiled binary-safe runtime
are connected to a complete real-world operator workflow.

The final verification chain is:

    P1-P12 reference semantics
    ->
    GLYPH PROOF GRAPH OK
    ->
    R0-R6 compiled runtime conformance
    ->
    GLYPH RUNTIME CONFORMANCE OK
    ->
    O1-O6 operator conformance
    ->
    GLYPH OPERATOR CONFORMANCE OK
    ->
    VERIFY OK

`VERIFY OK` is forbidden unless every required node passes.

## External dependencies

The operator graph requires two already-closed executable graphs:

- `P1-P12` — `GLYPH_PROOF_GRAPH_V1`;
- `R0-R6` — `GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1`.

The operator graph does not replace either graph.

## Nodes

### O1 — Deterministic Filesystem Manifest

Establishes:

- deterministic source discovery;
- raw path-byte ordering;
- stable document IDs;
- committed source payloads;
- ordered corpus identity;
- source mutation rejection.

Depends on:

- P1-P12;
- R0-R6.

### O2 — Deterministic Runtime Index

Establishes:

- one binary-safe compiled index per committed document;
- virtual logical sentinel 256;
- 257-symbol alphabet;
- runtime binary commitments;
- deterministic and atomically published runtime indexes.

Depends on O1.

### O3 — Binary Query and Source Mapping

Establishes:

- binary-safe query-file and query-hex transport;
- compiled FM count;
- compiled locate;
- canonical `(doc_id, doc_offset)` coordinates;
- committed filesystem path mapping;
- independent byte checks;
- document-boundary enforcement.

Depends on O2.

### O4 — Self-contained Operator Evidence Bundle

Establishes:

- complete source and runtime commitments;
- bundled query and artifact;
- bundled compiled binaries;
- exact manifest coverage;
- independent replay outside the repository;
- no network or external-data dependency.

Depends on O3.

### O5 — One-command Operator Workflow

Establishes:

- one command executes O1 through O4;
- deterministic case construction;
- query-file and query-hex equivalence;
- case verification after source removal;
- atomic final publication;
- independently replayable contained bundle.

Depends on O4.

### O6 — Operator Conformance Closure

O6 requires every preceding operator node and both external graphs.

It permits final `VERIFY OK` only when:

- O1 through O5 returned canonical successful gate results;
- every current mutation test was rejected;
- every specification, checker, and result artifact exists;
- dependency order is complete;
- no node is duplicated, omitted, reordered, failed, or skipped.

## Required sequence

The only accepted node sequence is:

    O1
    O2
    O3
    O4
    O5
    O6

## Required dependencies

    O1 -> P1-P12, R0-R6
    O2 -> O1
    O3 -> O2
    O4 -> O3
    O5 -> O4
    O6 -> O1, O2, O3, O4, O5, P1-P12, R0-R6

## Mutation gate

The executable graph must reject at least:

- a missing operator node;
- a duplicate operator node;
- a dependency-order violation;
- a failed operator node;
- an undeclared external dependency;
- a closure node that skips an obligation.

## Final invariant

The top-level verifier must print exactly one:

    VERIFY OK

It must occur strictly after:

    GLYPH PROOF GRAPH OK
    GLYPH RUNTIME CONFORMANCE OK
    GLYPH OPERATOR CONFORMANCE OK
