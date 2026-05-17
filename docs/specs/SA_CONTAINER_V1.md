# GLYPH SA Container v1

Status:

- planned
- not yet implemented
- intended as migration path from raw SA32 to versioned SA artifacts

---

## Purpose

Current `sa.bin` is a raw uint32 suffix array.

That format is temporary and unversioned.

SA Container v1 defines a future explicit artifact format for suffix arrays,
so GLYPH can distinguish:

- SA32 vs SA64
- endian assumptions
- entry width
- corpus size
- artifact version

This is required before introducing SA64.

---

## Why raw sa.bin cannot evolve safely

Current format:

    sa.bin = raw uint32 array

Problems:

- no magic bytes
- no version field
- no entry width
- no corpus byte length
- no endian marker
- no artifact type marker

A raw uint32 SA file and a future raw uint64 SA file cannot be safely
distinguished by readers without external metadata.

Therefore SA64 must not silently reuse raw `sa.bin`.

---

## Proposed container layout

Magic:

    GLYPHSA1

Header:

    offset  size  field
    --------------------------------
    0       8     magic = "GLYPHSA1"
    8       4     version = 1 (uint32)
    12      4     entry_width = 4 or 8 (uint32)
    16      8     corpus_bytes (uint64)
    24      8     sa_entries (uint64)
    32      4     endian = 1 for little-endian (uint32)
    36      4     reserved_flags (uint32)
    40      ...   suffix array entries

Entry encoding:

- entry_width = 4 → uint32 entries
- entry_width = 8 → uint64 entries

---

## Compatibility policy

Current GLYPH v0.x keeps writing:

    sa.bin

as raw uint32 for existing tools.

SA Container v1 should be introduced as a separate artifact:

    sa_v1.bin

Existing tools remain unchanged until container-aware readers exist.

Migration path:

1. keep raw `sa.bin`
2. add `sa_v1.bin` writer
3. add container-aware SA reader
4. update build_bwt to accept either raw SA32 or SA container
5. introduce SA64 as `GLYPHSA1` with `entry_width = 8`

---

## Safety requirements

SA container readers must validate:

- magic bytes
- version
- entry_width
- corpus_bytes > 0
- sa_entries == corpus_bytes
- file size matches header + entries
- all SA values are within `[0, corpus_bytes)`

Failure must be fail-fast.

No silent fallback from bad container to raw mode.

---

## Relationship to SA64

SA64 should be a format-compatible extension of the container:

    GLYPHSA1 + entry_width = 8

This avoids creating separate incompatible artifact families.

SA64 is then a data-width change, not a new undocumented file type.

---

## Status

This document defines the intended artifact contract.

Implementation is pending.
