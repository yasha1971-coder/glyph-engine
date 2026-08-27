# GLYPH Working State — 2026-08-27 V1

Status: active engineering record

Last checked: 2026-08-27

Authoritative baseline commit:
`c4138812f60704083f099fe0daeb6461abc9d0c2`

## Purpose

Record the exact project state, claim boundary, working discipline and next
decision so that later work does not silently replace measured history with
memory.

This file is not a product-readiness claim and is not a replacement for an
executable gate.

## Current movement

The active movement is:

    Composition V1 independent replay
    -> final composition reference marker
    -> unchanged P/R/O/I0 full verification
    -> review and one-purpose commit

The active movement is not runtime compression yet.

After Composition V1 closes, the next proposed movement is a separate
`RLBWT_BINARY_SAFE_V2` branch that ports the already measured compressed
query/locate path across the current binary-safe operator and evidence
boundary.

## Runtime and deployment separation

The OVH working branch is isolated at:

    ~/GLYPH_WORKTREES/composition-independent-replay-v1

The public demo continues to use its existing service and runtime profile:

    GLYPH_BINARY_RUNTIME_V1

Composition reference work must not:

- restart or modify the live demo;
- rewrite the live runtime index;
- change the Cloudflare Tunnel or public Worker;
- claim that Composition V1 is already part of top-level `VERIFY OK`;
- claim that the current demo uses RLBWT.

## Claim ledger

### WS-01 — Composition mutation closure at the baseline

CLAIM: The Composition V1 reference checker at the baseline maps all 25
normative mutation requirements as `EXACT`, with zero open requirements.

SCOPE: `tools/check_composition_reference_v1.py` at `c413881`.

STATUS: `MEASURED_REPORTED`.

EVIDENCE: OVH output reported
`normative_exact_count = 25`, `normative_open_count = 0`, 34 implemented
mutation tests and `GLYPH COMPOSITION REFERENCE NORMAL PATH OK`.

FALSIFIER: A clean baseline run produces a nonzero exit status, an open or
non-exact normative requirement, or a different mutation count.

SUPERSEDES: Earlier Composition V1 states with M01-M25 still open.

LAST_CHECKED: 2026-08-27.

### WS-02 — Existing required verification closure

CLAIM: The existing P1-P12, R0-R6, O1-O6 and Embedded I0 chain is green at
the baseline.

SCOPE: `./verify.sh` at `c413881` on OVH.

STATUS: `MEASURED_REPORTED`.

EVIDENCE: The reported clean run ended with all four graph markers and
`VERIFY OK`.

FALSIFIER: A clean rerun of `./verify.sh` fails or any required marker is
absent.

SUPERSEDES: None.

LAST_CHECKED: 2026-08-27.

### WS-03 — Live demo state

CLAIM: The loopback origin and public health route are ready and expose
`GLYPH_BINARY_RUNTIME_V1` with runtime index ID
`986f5f94eadb65439c3fb84eb77e75786e98a0e04687403d7b772607f67c42d9`.

SCOPE: The existing OVH demo deployment, not the composition worktree.

STATUS: `MEASURED_REPORTED`.

EVIDENCE: Reported `127.0.0.1:8787/health` and
`https://demo.glyph.rs/health` responses on 2026-08-27.

FALSIFIER: Either health route fails or reports a different runtime identity
without a corresponding deployment binding.

SUPERSEDES: Earlier design-only deployment state.

LAST_CHECKED: 2026-08-27.

### WS-04 — Current demo runtime size

CLAIM: The measured 50,000,000-byte demo runtime for
`GLYPH_BINARY_RUNTIME_V1` is 1,906,253,270 bytes.

SCOPE: The current dense, binary-safe reference runtime with checkpoint step
32; not an RLBWT lower bound.

STATUS: `MEASURED_REPORTED`.

EVIDENCE: `demo/v0/runtime_binding_v0.json` and its contract verification.

FALSIFIER: A bound deterministic rebuild of the same runtime identity
produces a different declared size.

SUPERSEDES: Any statement that treats this size as the minimum necessary
GLYPH query runtime.

LAST_CHECKED: 2026-08-27.

### WS-05 — Historical compressed runtime path

CLAIM: GLYPH previously measured an RLBWT exact count runtime below corpus
size and an RLBWT C++ full query+locate runtime near or below corpus size on
the recorded fixtures.

SCOPE: Historical RLBWT profiles, physical-sentinel-era fixtures and their
recorded benchmark conditions; not the current binary-safe demo runtime.

STATUS: `MEASURED_HISTORICAL`.

EVIDENCE:

- `benchmarks/results/RLBWT_QUERY_RUNTIME_PROFILE_V1_VERIFY.md` records
  0.891x on Pizza 50 MB and 0.401x on synthetic logs for query/count;
- `benchmarks/results/RLBWT_CPP_FULL_RUNTIME_VS_COMPACT_V1.md` records
  1.016x on Pizza 50 MB and 0.526x on synthetic logs for full query+locate.

FALSIFIER: The committed artifacts do not reproduce under their declared
inputs, profiles and measurement rules.

SUPERSEDES: The claim that 10x-40x runtime growth is a fundamental property
of GLYPH.

LAST_CHECKED: 2026-08-27.

### WS-06 — Why the sub-1x path is not the current demo path

CLAIM: The historical RLBWT profile was not carried across the later
binary-safe 257-symbol, multi-document, operator, evidence and composition
closure.

SCOPE: Repository architecture and history, not a performance measurement.

STATUS: `DERIVED_FROM_CODE_HISTORY`.

EVIDENCE: Current operator manifests bind `GLYPH_BINARY_RUNTIME_V1`, logical
sentinel 256 and alphabet size 257, while the RLBWT benchmark/profile files
remain separate research artifacts and are absent from the live runtime
binding.

FALSIFIER: A committed runtime profile is found that combines RLBWT count and
locate with the current binary-safe operator/evidence/composition chain and
passes the required closure.

SUPERSEDES: The question "where did RLBWT disappear?"; it remained in the
repository but was not promoted through the newer proof boundary.

LAST_CHECKED: 2026-08-27.

### WS-07 — Independent Composition V1 replay candidate

CLAIM: The candidate change adds a process-independent composition replay
that does not import the composition checker and that recomputes root,
runtime-unit coverage, global identities, exact count, bounded global
coordinates and returned source bytes.

SCOPE: The committed OVH candidate based on `c413881`.

STATUS: `MEASURED_OVH_COMMITTED_CANDIDATE`; remote review and promotion remain pending.

EVIDENCE: The OVH run retained under
`~/GLYPH_WORK_RECORDS/composition-independent-replay-v1/20260827T010323Z-3478234/`
produced two byte-identical targeted outputs with SHA256
`2295ac12e60f4155e79cc89c06fef7ad4059bc747dd53000d43b26c65795aa9a`.
The gate reported 25 normative requirements as `EXACT`, zero open
requirements and three rejected independent replay mutations. The unchanged
P/R/O/I0 chain ended with `VERIFY OK`, and loopback health was byte-identical
before and after validation.

FALSIFIER: A clean reproduction from the resulting commit produces
different targeted outputs, accepts a negative replay mutation, changes live
health or fails the existing required verification chain.

SUPERSEDES: The baseline state where 25/25 mutation closure existed but the
separate independent replay and final reference marker did not.

LAST_CHECKED: 2026-08-27.


### WS-08 — Path-sensitive RLBWT tiny-runtime size observation

CLAIM: The same tiny RLBWT fixture reported 5,755 runtime bytes in the
baseline repository path and 5,979 bytes in the isolated worktree, a
difference of 224 bytes.

SCOPE: The research-tier tiny RLBWT fixture; not Composition V1 and not the
live binary-safe demo runtime.

STATUS: `MEASURED_REPORTED`; root cause remains `HYPOTHESIS`.

EVIDENCE: OVH verification output from 2026-08-27. The working hypothesis is
that absolute filesystem paths are serialized into counted metadata.

FALSIFIER: Rebuilding the identical fixture under different absolute paths
produces byte-identical manifests and identical runtime totals.

SUPERSEDES: None. This observation does not block Composition V1 closure and
must be investigated separately in `RLBWT_BINARY_SAFE_V2`.

LAST_CHECKED: 2026-08-27.


## Mandatory working protocol

Every movement uses this order:

1. Resolve an exact clean base commit and create a dedicated worktree.
2. Record live-demo health before changes; do not modify the service.
3. Change one proof obligation or one architecture boundary only.
4. Run Python syntax checks for every changed Python entrypoint.
5. Run the targeted checker once and require its exact success marker.
6. Run the targeted checker again and require byte-identical output.
7. Run the unchanged full `./verify.sh` chain.
8. Run `git diff --check` and inspect the complete diff and final status.
9. Create one purpose-specific commit only after every gate is green.
10. Push or promote only after the commit identity and claim boundary are
    recorded.

OVH candidate validation logs are retained outside the repository under:

    ~/GLYPH_WORK_RECORDS/composition-independent-replay-v1/<UTC-run-id>/

The record contains pre/post live health, two targeted outputs, the full
verification log, diff checks and final Git status. These observational logs
do not participate in artifact identity.

If any step fails:

- stop the movement;
- do not commit partial closure;
- do not modify or restart the live demo;
- retain the failure output as evidence;
- classify the result as failed or open, never as verified.

## Gate markers for the current movement

The candidate is accepted only if one OVH run contains all of:

    GLYPH COMPOSITION INDEPENDENT REPLAY OK
    GLYPH COMPOSITION REFERENCE OK
    GLYPH PROOF GRAPH OK
    GLYPH RUNTIME CONFORMANCE OK
    GLYPH OPERATOR CONFORMANCE OK
    GLYPH EMBEDDED I0 CONTRACT VERIFY PASS
    VERIFY OK

The independent replay marker appears inside the targeted replay output. The
composition reference marker remains a research-tier gate and is not added to
top-level `verify.sh` by this movement.

## Explicit non-claims

This movement does not establish:

- a production Index Forest;
- distributed composition;
- arbitrary block pruning;
- field-aware routing;
- current-demo RLBWT deployment;
- sub-1x binary-safe full runtime;
- production readiness;
- replacement of Elasticsearch, Redis, Snowflake or vector databases.

## Next decision

The OVH gates passed on 2026-08-27 and the one-purpose candidate commit exists. The next action is isolated remote publication and review. Then make a
separate explicit decision whether Composition V1 should remain a
research-tier executable gate or be promoted into required top-level
verification.

Only after that decision start `RLBWT_BINARY_SAFE_V2`; do not mix compression
work into the Composition V1 closure commit.
