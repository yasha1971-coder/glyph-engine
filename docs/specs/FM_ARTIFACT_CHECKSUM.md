# GLYPH FM Artifact Checksum Design

Status: design locked, implementation pending.

---

## Problem

Current `fm.bin` uses an explicit magic string:

    FMBINv1\0

The current FM artifact has a versioned magic/header, but no checksum over
the checkpoint payload.

Current corruption detection path:

- manifest.json links corpus and artifacts
- FM binary is checked for existence
- FM magic is checked at load
- FM checkpoint data is not checksummed internally

A corrupted checkpoint array could pass current header checks.

---

## Current FMBINv1 layout

Current layout written by `src/build_fm.cpp`:

    offset  size   field
    --------------------------------
    0       8      magic = "FMBINv1\0"
    8       8      n (uint64)
    16      4      checkpoint_step (uint32)
    20      8      num_blocks (uint64)
    28      2048   C[256] uint64 table
    2076    ...    checkpoints[num_blocks][256] uint32

Version is encoded in the magic string.

There is no separate `version` field.

---

## Why FMBINv2, not a patch to v1

Appending a checksum to FMBINv1 would be ambiguous:

- old readers may ignore trailing bytes
- readers cannot distinguish plain v1 from "v1 plus checksum"
- the same magic would describe two layouts

Decision:

    introduce FMBINv2

New readers should reject v1 when v2 is required.
Old readers should reject v2 via magic mismatch.

No silent fallback.

---

## Checksum algorithm

Candidates:

- FNV-1a: simple, no dependency, slower
- CRC32C: fast with hardware support, hardware-dependent
- xxHash64: fast software hash, widely used, 64-bit output

Initial implementation decision:

    FNV-1a 64-bit

Reason:

- no external dependency
- tiny implementation
- stable across platforms
- sufficient for corruption detection
- low implementation risk

Future optimization:

    xxHash64 may replace FNV-1a in a later format version
    if checksum time becomes measurable.

This checksum is not a security mechanism.
It is an artifact corruption detector.

---

## What the checksum covers

Decision:

    checkpoints payload only

Covered bytes:

    checkpoints[num_blocks][256] uint32

Not covered:

- magic
- n
- checkpoint_step
- num_blocks
- C[256]

Reason:

The checkpoint payload is the large corruption-prone data region.
Header fields are already parsed and validated structurally.

Future formats may choose to checksum the full post-magic payload, but v2
keeps the scope minimal and explicit.

---

## Proposed FMBINv2 layout

    offset  size   field
    --------------------------------
    0       8      magic = "FMBINv2\0"
    8       8      n (uint64)
    16      4      checkpoint_step (uint32)
    20      8      num_blocks (uint64)
    28      2048   C[256] uint64 table
    2076    8      checkpoint_payload_bytes (uint64)
    2084    8      checkpoint_xxhash64 (uint64)
    2092    ...    checkpoints[num_blocks][256] uint32

Expected:

    checkpoint_payload_bytes == num_blocks * 256 * sizeof(uint32)

Readers must validate:

- magic == FMBINv2\0
- checkpoint_step > 0
- num_blocks > 0
- checkpoint_payload_bytes matches expected size
- file contains exactly the expected checkpoint payload
- xxHash64(checkpoints) matches checkpoint_xxhash64

---

## Backward compatibility

FMBINv1:

- current format
- no internal checksum
- still documented

FMBINv2:

- checksum-bearing format
- required for future strict artifact verification

Migration:

- rebuild FM artifacts with updated build_fm
- keep old v1 docs for historical reference
- do not silently accept v1 in strict mode

---

## Implementation plan

1. Vendor or implement xxHash64 without external build dependency.
2. Update `src/build_fm.cpp`:
   - write magic `FMBINv2\0`
   - compute xxHash64 over checkpoint payload
   - write `checkpoint_payload_bytes`
   - write `checkpoint_xxhash64`
3. Update FM readers:
   - `src/query_fm_v1.cpp`
   - `src/query_fm_server_v1.cpp`
   - `src/query_fm_batch_v1.cpp`
4. Add tests:
   - valid FMBINv2 loads correctly
   - bad magic rejected
   - truncated payload rejected
   - corrupted checkpoint byte rejected
   - payload byte count mismatch rejected
5. Rebuild generated FM artifacts:
   - examples/mini
   - bench_1gb when needed

---

## Relationship to Query Tiers

Tier 1:

    may use trusted already-loaded FMBINv2

Tier 2:

    uses manifest-level verification

Tier 3:

    requires internal artifact checksum validation

FMBINv2 is a prerequisite for Tier 3.
