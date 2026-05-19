# GLYPH Query Protocol V1 — Status

Status:
    IMPLEMENTED

Commit:
    8144208

Capabilities:
    - deterministic JSON query output
    - stable interval/count contract
    - exact-match FM retrieval
    - corruption-verified FMBINv2 artifacts
    - CI regression coverage

CLI:

    query_fm_v1 <fm.bin> <bwt.bin> <pattern_hex> --json

Example output:

{
  "pattern_hex": "6572726f72",
  "interval": [20, 22],
  "count": 2,
  "fm_version": "FMBINv2",
  "verified": true
}

Determinism invariants:
    - no timestamps
    - no random IDs
    - no environment-dependent fields
    - byte-stable JSON output

Role in architecture:
    GLYPH Query Protocol V1 is the canonical machine-readable
    exact retrieval interface for local deterministic corpora.

Future extensions:
    - locate payloads
    - shard-aware responses
    - HTTP protocol bridge
    - persistent backend integration
