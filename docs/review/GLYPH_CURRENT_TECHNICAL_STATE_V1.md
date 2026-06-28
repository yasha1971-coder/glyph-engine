# GLYPH_CURRENT_TECHNICAL_STATE_V1

Status: current technical state map  
Date: 2026-06-27

## Purpose

Record the current technical state of GLYPH after the RLBWT bounded evidence review checkpoint and the first binary-safe reference work.

This document separates:

- verified working paths
- reference-only paths
- known boundaries
- explicit non-claims
- implementation triggers

It is intended to prevent overclaiming and to guide the next engineering steps.

## Current verified working path

The currently verified working path is:

    RLBWT bounded evidence over sentinel-safe fixed corpora

Review checkpoint:

    tag: rlbwt-bounded-evidence-v1

Checkpoint commit:

    d9ae6b0 docs: harden RLBWT bounded evidence public wording

Primary reviewer entrypoint:

    docs/review/RLBWT_BOUNDED_EVIDENCE_REVIEW_PATH_V1.md

One-command verification:

    ./verify.sh

Expected high-level result:

    [tiny-fixture] PASS
    VERIFY OK

## What the verified path currently demonstrates

The verified RLBWT bounded evidence path demonstrates:

    fixed corpus
    -> sentinel-safe GLYPH index
    -> RLBWT full runtime
    -> exact query
    -> FM interval
    -> exact match_count
    -> bounded offsets
    -> byte_check
    -> evidence artifact
    -> artifact replay verifier
    -> portable bundle
    -> bundle replay verifier
    -> schema smoke validation
    -> one-command verification

## Verified artifact and bundle versions

Artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

Bundle version:

    RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1

Tarball version:

    RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1

Runtime profile:

    RLBWT_FULL_RUNTIME_PROFILE_V1

## Current verified tiny fixture

Current tiny fixture corpus:

    alpha beta gamma
    the quick brown fox
    delta the epsilon
    the final line

Query:

    the

Verified result:

- FM interval: [65, 68]
- match_count: 3
- max_offsets: 2
- returned_count: 2
- offsets: [43, 55]
- bounded: true
- byte_check: true
- artifact replay: PASS
- bundle replay: PASS
- schema validation: PASS

## Current larger measured results

The repository also contains measured local reports for larger corpora.

Examples include:

- Pizza 50MB
- XZ CVE corpus
- synthetic logs 50MB

Important measured properties include:

- persistent RLBWT server warm query latency
- high-count locate stress
- bounded locate mode
- bounded evidence artifact creation
- artifact replay
- bundle replay
- tarball replay

These reports are useful as engineering evidence, but the public review checkpoint is intentionally centered on a tiny reproducible fixture.

## Current binary-safe status

Current production GLYPH v0.x path is not yet binary-safe.

Current v0.x invariant:

    source corpus must not contain 0x00
    index corpus = source corpus + physical 0x00 sentinel

This boundary is documented in:

    docs/specs/BINARY_SAFE_ROADMAP_V1.md
    benchmarks/results/BINARY_SAFE_CURRENT_BOUNDARY_V1.md

Boundary probe:

    tools/run_binary_safe_boundary_probe_v1.sh

Current boundary result:

    BOUNDARY_PROBE_EXPECTED_FAIL

This expected failure means the current sentinel-safe builder rejects or fails on source corpora containing 0x00.

## Binary-safe design decision

Binary-safe target design is documented in:

    docs/specs/BINARY_SAFE_DESIGN_DECISION_V1.md

Chosen direction:

    virtual sentinel model

Data alphabet:

    0..255

Internal boundary symbol:

    256

Key invariant:

    match_count == number of real non-wrapping occurrences in the source corpus

Every returned offset must satisfy:

    source_corpus[offset : offset + query_len] == query_bytes

The design rejects plain cyclic BWT + primary_index only as sufficient evidence semantics, because boundary-crossing false matches could corrupt exact match_count.

## Binary-safe reference-only work

The following binary-safe components are reference/prototype work only.

They do not mean the production C++ runtime is binary-safe.

### Binary-safe symbol corpus prototype

Commit:

    8d4e5b6 core: add binary-safe symbol corpus prototype

Tools:

    tools/make_binary_safe_symbol_corpus_v1.py
    tools/verify_binary_safe_symbol_corpus_v1.py

Validated corpus bytes:

    41 00 42 00 43 00 42

Validated symbol stream:

    65, 0, 66, 0, 67, 0, 66, 256

Validated invariants:

- source bytes: 7
- NUL bytes: 3
- symbols: 8
- 0x00 preserved as data symbol 0
- virtual sentinel is 256
- exactly one sentinel
- sentinel is last
- all source symbols are 0..255

### Binary-safe FM tiny reference fixture

Commit:

    954860f core: add binary-safe FM tiny reference fixture

Tool:

    tools/run_binary_safe_fm_tiny_fixture_v1.py

Corpus bytes:

    41 00 42 00 43 00 42

Query bytes:

    00 42

Reference result:

- FM interval: [0, 2]
- match_count: 2
- offsets: [1, 5]
- byte_check: true
- rejected boundary-crossing offsets: []

This is the reference behavior that a future binary-safe C++ runtime must match.

## Explicit non-claims

GLYPH currently does not claim:

- production binary-safe indexing
- arbitrary malware/RAM dump support
- distributed search
- petabyte-scale deployment
- timestamped evidence
- signed evidence
- notarized evidence
- court-grade legal proof
- source authenticity beyond recorded hashes
- exhaustive locate when bounded=true
- semantic search
- fuzzy search
- token search
- ranking
- SIEM replacement
- Elasticsearch replacement
- Splunk replacement
- VirusTotal replacement
- zero-knowledge proof
- authenticated non-membership proof

## Current strongest technical claim

The current strongest technical claim is:

    GLYPH has a reproducible review checkpoint for replayable exact-byte bounded evidence over fixed sentinel-safe corpora.

More specifically:

    fixed corpus
    + compressed RLBWT runtime
    + exact FM interval/count
    + bounded offsets
    + byte_check
    + artifact replay
    + portable bundle replay
    + schema smoke validation
    + one-command verifier

## Current binary-safe claim

The current binary-safe claim is limited to:

    GLYPH has documented the current 0x00 boundary,
    selected a virtual sentinel design,
    and implemented positive reference fixtures showing how 0x00 can be preserved as data while sentinel is represented as symbol 256.

It does not claim:

    full binary-safe production runtime

## Implementation triggers

Full binary-safe runtime should be implemented when there is a concrete external need, such as:

- a real DFIR/malware/log user needs 0x00-containing corpora
- a reviewer asks to test binary corpora
- a real corpus/pattern scenario is provided
- a target segment requires binary-safe evidence artifacts

Until then, binary-safe work remains a documented boundary plus reference semantics.

## Next reasonable engineering options

Possible next steps:

1. Add this technical state document to README/review path.
2. Add binary-safe status to public review docs.
3. Add a lightweight GLYPHBench / evidence-chain comparison document.
4. Add binary-safe runtime only after concrete demand.
5. Add timestamp/signature only after legal/compliance demand.
6. Add multi-corpus manifest only after multi-corpus demand.

## Current recommended engineering stance

Do not overbuild.

Keep the current verified RLBWT bounded evidence checkpoint stable.

Use binary-safe reference fixtures as a prepared technical answer, not as a reason to build the full binary-safe runtime immediately.

The next major implementation should be triggered by a real user case, not by theoretical completeness.
