# GLYPH Query Protocol V1

Status: draft

Purpose:

Stable deterministic machine-readable query output for agents, tooling,
benchmarks, and HTTP integration.

## JSON output example

```json
{
  "pattern_hex": "6572726f72",
  "interval": [20, 22],
  "count": 2,
  "fm_version": "FMBINv2",
  "verified": true
}
Required fields

* pattern_hex
* interval
* count
* fm_version
* verified

Determinism rule

No timestamps.
No random IDs.
No environment-dependent fields.

Identical corpus + identical artifacts + identical query must produce
byte-identical JSON output.
