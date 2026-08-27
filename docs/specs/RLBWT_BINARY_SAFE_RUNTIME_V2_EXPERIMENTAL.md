# RLBWT Binary-safe Runtime V2 — Experimental Contract

## Status

This is an additive experimental runtime contract. It does not redefine RLB1,
RLR1, RLB2, LOC1, or any existing production runtime.

## Runtime set

A manifest-bound runtime contains exactly three data artifacts:

1. one `GLYPH_RLB2_EXPERIMENTAL_V2` run container;
2. one `GLYPH_RLR2_V2` rank index;
3. one `LOC1` locate-sample file.

A canonical JSON manifest binds those artifacts into one runtime identity.

## Canonical manifest

The manifest format is `GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2`, version `1`.

It records:

- a safe reference identifier;
- corpus byte length, MD5, and SHA-256;
- BWT row count;
- rank step;
- locate sample step;
- exact runtime-data byte total;
- exact role, basename, format, byte length, and SHA-256 for every artifact.

JSON must be canonical: sorted keys, compact separators, UTF-8, and one final
newline. Duplicate keys, unknown keys, missing keys, unsafe basenames, and
noncanonical serialization are rejected.

The manifest may describe any valid corpus identity. A benchmark-specific
size advantage is evidence, not a condition of format validity.

## Cross-artifact binding

The runtime must fail closed unless:

- the RLR2 source byte length and SHA-256 match the selected RLB2;
- RLB2 and RLR2 row counts agree;
- manifest rank and sample steps agree with their artifacts;
- LOC1 SA size equals the runtime row count;
- every artifact size and SHA-256 matches the manifest;
- the exact sum of the three artifact sizes equals `runtime_data_bytes`.

Artifacts and the manifest must be regular, non-symlink files and stable
during identity inspection.

## Query semantics

Patterns are non-empty byte strings expressed as hexadecimal input.

For a valid pattern, the runtime returns:

- the exact FM interval;
- exact match count;
- zero or more sorted corpus offsets;
- whether returned offsets are complete;
- LF-step accounting.

`max_offsets = -1` requests all offsets. Values below `-1` are invalid.

The logical sentinel is internal runtime state and is not an ordinary query
byte. An ordinary byte pattern must never resolve to the terminal suffix as a
reported corpus match.

Rank inside a run includes the prefix from the checkpoint position to the
queried position. Omitting that intra-run prefix is a correctness failure.

## Binding modes

Canonical runtime mode accepts one manifest path.

A research-only explicit mode accepts all three artifact paths. The two modes
cannot be mixed, and partial explicit binding is invalid.

Size and reproducibility claims must use canonical manifest mode.

## Fail-closed requirements

The implementation rejects:

- malformed, duplicate-key, or noncanonical manifests;
- incompatible versions or formats;
- unsafe identities or filenames;
- invalid corpus hashes or geometry;
- missing, extra, mutated, truncated, or identity-mismatched artifacts;
- mixed or incomplete binding modes;
- malformed or empty patterns;
- invalid locate limits.

## Claim scope

The enwik8 sub-1x result is scoped to the exact corpus and artifact hashes in
the corresponding benchmark report. This specification alone makes no
universal compression-ratio or performance claim.
