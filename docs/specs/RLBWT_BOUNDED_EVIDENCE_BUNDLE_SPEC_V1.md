# RLBWT_BOUNDED_EVIDENCE_BUNDLE_SPEC_V1

Status: draft  
Date: 2026-06-27

## Purpose

Define the portable bundle format for GLYPH RLBWT bounded evidence.

Bundle version:

    RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1

Contained artifact version:

    RLBWT_BOUNDED_EVIDENCE_V1

Runtime profile:

    RLBWT_FULL_RUNTIME_PROFILE_V1

Primary goal:

    package everything needed to replay bounded exact evidence
    over compressed RLBWT runtime
    without depending on the original local /tmp paths

## Bundle layout

A V1 bundle is a directory with this minimum structure:

    bundle/
      bundle_manifest_v1.json
      README_REPLAY.md
      corpus.bin
      evidence.json
      runtime/
        bwt.rlbwt
        bwt.rlbwt.rank
        locate_core_s128.bin
        manifest.json
        rlbwt_full_runtime_manifest_v1.json

## Required files

### bundle_manifest_v1.json

The bundle manifest records:

- bundle version
- artifact version
- runtime profile
- evidence artifact path
- source corpus path
- runtime directory path
- file list
- file byte lengths
- file SHA256 hashes
- replay command

### README_REPLAY.md

Human-readable replay note.

It must explain:

- what the bundle contains
- how to replay it
- that bounded evidence is not exhaustive locate when `bounded=true`

### corpus.bin

The source corpus used for byte verification.

The corpus is required because byte checks verify returned offsets against source bytes.

### evidence.json

The bundled bounded evidence artifact.

This is the original `RLBWT_BOUNDED_EVIDENCE_V1` artifact with paths rewritten to bundle-relative paths:

- source corpus path becomes `corpus.bin`
- runtime dir becomes `runtime`
- runtime file paths become `runtime/<file>`

### runtime/

The compressed RLBWT runtime files required for replay.

Required files:

- `bwt.rlbwt`
- `bwt.rlbwt.rank`
- `locate_core_s128.bin`
- `manifest.json`
- `rlbwt_full_runtime_manifest_v1.json`

## Bundle manifest required fields

A V1 `bundle_manifest_v1.json` must contain:

- `bundle_version`
- `artifact_version`
- `profile`
- `evidence`
- `source_corpus`
- `runtime_dir`
- `files`
- `replay_command`

Each entry in `files` must contain:

- `path`
- `bytes`
- `sha256`

## Replay rules

A bundle verifier must check:

1. `bundle_manifest_v1.json` exists.
2. `bundle_version` is `RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1`.
3. Every file listed in `files` exists.
4. Every listed file byte length matches the manifest.
5. Every listed file SHA256 matches the manifest.
6. `evidence.json` exists.
7. Bundle-relative evidence paths are resolved to absolute local paths.
8. The resolved evidence artifact is passed to the RLBWT bounded evidence replay verifier.
9. The underlying bounded evidence replay verifier must pass.
10. Bundle replay is PASS only if both bundle-level and artifact-level checks pass.

## Artifact replay delegation

The bundle verifier does not duplicate all evidence semantics.

It delegates artifact semantics to:

    tools/verify_rlbwt_bounded_evidence_v1.py

That verifier checks:

- artifact version
- runtime profile
- query hex/hash
- source corpus size/hash
- runtime file size/hash
- recomputed FM interval
- recomputed match count
- recomputed bounded offsets
- byte checks for returned offsets

## Bounded semantics

The bundle does not change bounded evidence semantics.

The evidence artifact must preserve:

    match_count = full exact count

The returned offsets may be bounded:

    returned_count <= match_count

If:

    returned_count < match_count

then:

    bounded = true

The bundle must not claim exhaustive offset enumeration when `bounded=true`.

## Non-claims

A V1 bundle does not claim:

- exhaustive locate when `bounded=true`
- semantic search
- ranking
- fuzzy matching
- token search
- legal proof by itself
- authenticity of external source origin beyond recorded hashes
- compatibility with future runtime formats

## Current verified implementation

Bundle maker:

    tools/make_rlbwt_bounded_evidence_bundle_v1.py

Bundle verifier:

    tools/verify_rlbwt_bounded_evidence_bundle_v1.py

Current measured report:

    benchmarks/results/RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1.md

Current tiny verified replay result:

- query: `the`
- FM interval: [65, 68]
- match_count: 3
- max_offsets: 2
- returned_count: 2
- bounded: true
- offsets: [43, 55]
- byte_check: true
- bundle replay: PASS

## Future extensions

Future versions may add:

- tar/zip archive packaging
- JSON schema
- embedded bundle signature
- source manifest hash
- engine commit field
- server protocol version field
- C++ bundle verifier
- integration with existing GLYPH Evidence Bundle V1 family
