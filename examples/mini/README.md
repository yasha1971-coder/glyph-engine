# Mini example

This is a minimal end-to-end GLYPH pipeline.

It builds a tiny corpus, constructs:

- SA32u
- BWT
- FM-index

using the canonical sentinel-safe builder pipeline.

Then it runs a direct FM query for `error`.

Run:

    ./examples/mini/build_mini.sh

Expected output includes:

    count:    2

This example is intentionally small and does not use the HTTP layer.

The mini example validates the same canonical FM-index invariant used by the segmented v0.2 pipeline.

For the larger local benchmark, see:

- RUNBOOK_4GB.md

## Audit Artifact V0

The mini example can also generate and verify a portable audit artifact.

This artifact records:

- corpus hash
- index manifest hash
- query hash
- FM interval
- match count
- replay command
- reproduce status

It is not a legal proof, zero-knowledge proof, or production cryptographic proof system.

It is a minimal reproducible record for exact retrieval over a fixed committed corpus.

Build the mini index:

    ./examples/mini/run_mini.sh

Create an audit artifact:

    python3 tools/glyph_make_audit_artifact_v0.py \
      --index-dir examples/mini/out \
      --query error \
      --output examples/mini/out/audit_artifact_v0.json

Verify the audit artifact:

    python3 tools/glyph_verify_audit_artifact_v0.py \
      examples/mini/out/audit_artifact_v0.json

Expected output includes:

    VERIFY AUDIT ARTIFACT OK

Generated artifacts under `examples/mini/out/` are local build outputs and should not be committed.
