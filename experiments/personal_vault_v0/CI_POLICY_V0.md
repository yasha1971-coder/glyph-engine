# GLYPH Personal Vault CI Policy V0

Status: active policy for the Personal Vault research branch. This document changes no Runtime V2 format and no canonical Vault data.

## Purpose

Keep product health, storage correctness, and open research visibly separate.

A failed research hypothesis must remain reproducible evidence, but it must not make a stable product path appear broken. Conversely, a research workflow must never be made green by suppressing a genuine failure.

## Product CI

Product-facing pull-request checks are fail-closed. Success means the tested contract actually executed and passed.

Primary focused gate:

- `product-mvp-v0.1`

It exercises the current minimum usable storage contract:

- repository init;
- folder ingest without source deletion;
- immutable segment publication;
- manifest/root verification;
- bit-perfect restore and SHA-256 equality;
- heterogeneous Real Intake V1;
- additive multi-segment growth;
- exact-search substrate across committed segments.

The broader `ci` and `personal-vault-v0-closed-loop` workflows remain correctness/regression gates while this branch is experimental.

## Research CI

Rejected, exploratory, model-dependent, expensive, or benchmark-frontier workflows are manual-only through `workflow_dispatch` and are named with a `research-` prefix.

Current examples include:

- local LLM planner experiments;
- constrained edge-agent experiments;
- cognitive-router experiments;
- semantic candidate-discovery baselines;
- maximal semantic retrieval experiments.

A manual research run is allowed to fail. Failure records a rejected or unresolved hypothesis; it is not product-health status.

## Evidence rule

No workflow is considered evidence merely because GitHub displays `success`.

A product gate must:

1. execute the intended program;
2. fail on any command error;
3. validate its machine-readable artifact when one is expected;
4. enforce the claimed acceptance condition;
5. only then return success.

Shell pipelines that can mask failures must use `set -o pipefail` or an equivalent explicit status check.

## Preservation rule

Historical failed experiments, commits, scripts, and artifacts are not deleted merely to make the repository look green. They remain part of the research record unless the owner explicitly authorizes destructive removal.

The cleanup performed by this policy changes automatic triggers and labels only; it does not rewrite history or erase negative results.

## Product/research boundary

Semantic recall and autonomous LLM planning are not requirements for GLYPH V0.1 product readiness.

They may later improve the `REMEMBER` layer, but they cannot mutate or weaken the canonical `STORE / VERIFY / RESTORE` contract.

## Promotion rule

A research feature can move into product CI only after:

- its contract is explicitly defined;
- its acceptance threshold is fixed before evaluation;
- at least one reproducible passing run exists;
- failure cannot corrupt canonical Vault state;
- the product gate remains fail-closed.
