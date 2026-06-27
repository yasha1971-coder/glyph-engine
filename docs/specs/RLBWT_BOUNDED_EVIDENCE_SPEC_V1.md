# RLBWT_BOUNDED_EVIDENCE_SPEC_V1

Status: draft  
Date: 2026-06-27

## Purpose

Define the minimum structure and replay invariants for bounded evidence artifacts produced from GLYPH RLBWT compressed runtime.

Artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

Runtime profile:

    RLBWT_FULL_RUNTIME_PROFILE_V1

Primary goal:

    exact query/count
    + deterministic bounded offsets
    + byte verification
    + replay verification

This spec describes a bounded evidence artifact, not a full exhaustive locate result.

## Runtime model

The artifact is produced from an RLBWT full runtime directory containing:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`
- `manifest.json`
- `rlbwt_full_runtime_manifest_v1.json`

The deployed runtime must not require:

- `bwt.bin`
- `fm.bin`
- `fm_core.bin`
- `sa.bin`
- `corpus.sentinel.bin`
- `chunk_map.bin`

The source corpus is still required for byte verification of returned offsets.

## Server protocol

The bounded server request format is:

    query_hex<TAB>max_offsets

The bounded server response format is:

    OK<TAB>l<TAB>r<TAB>match_count<TAB>located_count<TAB>bounded<TAB>total_steps<TAB>max_steps<TAB>offsets

Fields:

- `l`: FM interval left boundary
- `r`: FM interval right boundary
- `match_count`: full exact count, equal to `r - l`
- `located_count`: number of offsets returned
- `bounded`: true when `located_count < match_count`
- `total_steps`: LF steps used for returned offsets only
- `max_steps`: maximum LF steps for any returned offset
- `offsets`: comma-separated recovered corpus offsets

## Artifact required fields

A V1 artifact must contain:

### Artifact identity

- `artifact_version`
- `profile`
- `server_protocol`
- `ok`

### Engine identity

- repo/tool identity
- server binary identity

### Query identity

- query text
- query hex
- query SHA256
- query byte length

### Source corpus identity

- path
- byte length
- SHA256

### Runtime identity

For each runtime file:

- name
- path
- byte length
- SHA256

Required runtime files:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`
- `manifest.json`
- `rlbwt_full_runtime_manifest_v1.json`

### Retrieval result

- FM interval
- full match count
- total possible count
- max offsets
- returned count
- bounded flag
- returned offsets
- total LF steps for returned offsets
- max LF steps for returned offsets

### Byte check

For every returned offset:

- offset
- expected query hex
- observed corpus bytes as hex
- boolean match result

The artifact must contain an aggregate boolean:

- `all_returned_offsets_match_query`

## Replay invariants

A replay verifier must check:

1. Artifact version is `RLBWT_BOUNDED_EVIDENCE_V1`.
2. Runtime profile is `RLBWT_FULL_RUNTIME_PROFILE_V1`.
3. Query hex matches query text.
4. Query SHA256 matches query bytes.
5. Source corpus byte length matches artifact.
6. Source corpus SHA256 matches artifact.
7. Runtime file byte lengths match artifact.
8. Runtime file SHA256 hashes match artifact.
9. Server replay returns the same FM interval.
10. Server replay returns the same full match count.
11. Server replay returns the same returned count.
12. Server replay returns the same bounded flag.
13. Server replay returns the same bounded offsets.
14. Returned offsets byte-check against the source corpus.
15. Replay result is PASS only if all checks pass.

## Bounded semantics

Bounded evidence separates exact search from exhaustive locate.

The artifact must preserve the full exact count:

    match_count = r - l

But it may return fewer offsets:

    returned_count <= match_count

When:

    returned_count < match_count

then:

    bounded = true

When:

    returned_count == match_count

then:

    bounded = false

A count-only result is represented as:

    max_offsets = 0
    returned_count = 0
    offsets = []
    bounded = true if match_count > 0

## Non-claims

A V1 bounded evidence artifact does not claim:

- exhaustive offset enumeration when `bounded=true`
- semantic relevance
- ranking
- fuzzy matching
- token-level search
- legal proof by itself
- source authenticity beyond recorded hashes
- portability unless paths/files are packaged separately

## Current verified fixture

The tiny fixture is:

    tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh

It is included in top-level:

    ./verify.sh

The fixture validates:

    tiny corpus
    -> sentinel-safe GLYPH index
    -> RLBWT full runtime export
    -> bounded evidence artifact
    -> replay verifier
    -> byte_check

Current tiny fixture result:

- query: `the`
- FM interval: [65, 68]
- match_count: 3
- max_offsets: 2
- returned_count: 2
- bounded: true
- offsets: [43, 55]
- byte_check: true
- replay: PASS

## Future extensions

Future versions may add:

- portable bundle format
- schema file
- embedded runtime manifest hash
- source manifest hash
- paginated locate
- offset-window locate
- server protocol version field
- C++ replay verifier
- integration with Audit Artifact V0 / Evidence Case V1
