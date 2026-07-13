# GLYPH_VERIFY_CHAIN_V1

Status: normative draft  
Version: 1  
Proof obligation: P12  
Date: 2026-07-13

## Purpose

Define the final executable closure of GLYPH proof obligations P1 through P12.

P12 introduces no new retrieval semantics. It proves that all previously
defined obligations exist, execute, satisfy their dependencies, and pass before
the repository may print `VERIFY OK`.

## Core invariant

`VERIFY OK` is permitted only if:

    P1 PASS
    P2 PASS
    P3 PASS
    P4 PASS
    P5 PASS
    P6 PASS
    P7 PASS
    P8 PASS
    P9 PASS
    P10 PASS
    P11 PASS
    P12 PASS

No proof obligation may be omitted, duplicated, reordered past an unmet
dependency, replaced by a missing checker, or represented by a checker that
returns failure.

## Executable closure

The authoritative executable entrypoint is:

    tools/run_glyph_proof_graph_v1.sh

It invokes:

    tools/check_verify_chain_v1.py

The checker must:

- contain exactly P1 through P11 as prerequisite nodes;
- execute every prerequisite checker;
- reject any non-zero checker exit code;
- reject `ok != true` when a checker emits JSON;
- require `p12_ready = true` from P11;
- validate every specification and checker path;
- validate dependency ordering;
- reject missing or duplicate graph nodes;
- emit P12 PASS only after P1 through P11 pass;
- emit exactly twelve ordered PASS results.

## VERIFY integration

`verify.sh` must invoke the executable proof graph before its final
`VERIFY OK` line.

Because `verify.sh` uses `set -e`, any proof-graph failure terminates
verification before `VERIFY OK` can be printed.

## Failure semantics

The following must prevent `VERIFY OK`:

- missing proof node;
- missing specification;
- missing checker;
- checker failure;
- malformed checker result;
- wrong proof identity;
- unmet dependency;
- P11 without `p12_ready = true`;
- fewer or more than twelve proof results;
- any result other than PASS;
- proof-graph runner omission from `verify.sh`.

## P12 result

A valid result contains:

    proof_count = 12
    passed = 12
    failed = 0
    all_required_nodes_present = true
    all_dependencies_satisfied = true
    p11_handoff_accepted = true
    verify_ok_permitted = true

## Non-claims

P12 establishes executable engineering closure over the declared GLYPH proof
obligations. It does not establish legal truth, semantic truth of source
content, publisher identity, or completeness of an unknown external corpus.
