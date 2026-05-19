GLYPH QUERY BATCH PROTOCOL V1

Purpose

Deterministic batch exact retrieval.

Single request.
Multiple patterns.
Deterministic output.

Input

{
  "patterns":[
    "6572726f72",
    "343034"
  ]
}

Required result fields

pattern_hex
interval
count
verified

Global fields

fm_version

Rules

No timestamps.

No random IDs.

No environment-dependent fields.

Identical corpus +
identical artifacts +
identical patterns

must produce

byte-identical JSON.

Example output

{
  "results":[
    {
      "pattern_hex":"6572726f72",
      "interval":[20,22],
      "count":2,
      "verified":true
    },
    {
      "pattern_hex":"343034",
      "interval":[16,17],
      "count":1,
      "verified":true
    }
  ],
  "fm_version":"FMBINv2"
}
