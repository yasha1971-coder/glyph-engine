# RLBWT_BOUNDED_EVIDENCE_SCHEMA_V1

Status: local schema smoke validation  
Date: 2026-06-27

## Purpose

Add JSON schemas and dependency-free smoke validation for RLBWT bounded evidence artifacts and bundle manifests.

Schemas:

- `docs/schemas/RLBWT_BOUNDED_EVIDENCE_SCHEMA_V1.json`
- `docs/schemas/RLBWT_BOUNDED_EVIDENCE_BUNDLE_MANIFEST_SCHEMA_V1.json`

Validator:

- `tools/validate_rlbwt_bounded_evidence_schemas_v1.py`

## Validation target

Validated against the tiny fixture outputs:

- artifact: `examples/rlbwt-bounded-evidence-tiny/out/rlbwt_bounded_evidence_v1.json`
- bundle manifest: `examples/rlbwt-bounded-evidence-tiny/out/rlbwt_bounded_evidence_bundle_v1/bundle_manifest_v1.json`

## Checks

The smoke validator checks:

- schema JSON files parse
- artifact version
- runtime profile
- query identity fields
- source corpus identity fields
- required runtime file identities
- FM interval shape
- match_count equals `r - l`
- returned_count equals number of offsets
- byte_check is true
- bundle manifest version
- bundle file list
- bundle file SHA256 field shape

## Result

Schema smoke validation passed on the current tiny fixture artifact and bundle manifest.

## Verify integration

Schema smoke validation is now exercised through the tiny fixture runner:

    tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh

Because this fixture is invoked by top-level:

    ./verify.sh

the one-command verification path now covers:

    bounded evidence artifact replay
    portable bundle replay
    schema smoke validation for artifact
    schema smoke validation for bundle manifest

