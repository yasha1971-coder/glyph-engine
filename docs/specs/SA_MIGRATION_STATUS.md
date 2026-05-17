# SA Migration Status

Status: active migration bridge
Date: 2026-05-17

## Current State

Legacy pipeline:

    raw corpus
        -> raw uint32 sa.bin
        -> build_bwt
        -> build_fm

Current production compatibility:

- build_bwt expects raw uint32 SA
- existing pipeline remains unchanged
- no runtime migration yet

## Completed

### SA Container Specification

File:

    docs/specs/SA_CONTAINER_V1.md

Defined:

- magic
- version
- entry width
- corpus size
- endian flag
- reserved flags
- payload layout

### SA Container Writer

File:

    tools/write_sa_container_v1.py

Capabilities:

- wraps raw SA into GLYPHSA1 container
- validates:
  - empty file
  - entry width
  - divisibility
  - corpus/entry mismatch

### SA Container Reader

File:

    tools/read_sa_container_v1.py

Capabilities:

- validates container header
- validates:
  - magic
  - version
  - entry_width
  - file_size
- prints header metadata
- fail-fast on corruption

## Regression Coverage

Green tests:

- FM correctness tests
- locate tests
- manifest integrity tests
- verified query tests
- SA container writer tests
- SA container reader tests

Total:

    38 tests green

## Architectural Boundary Reached

Before:

    SA = anonymous binary blob

Now:

    SA = versioned artifact contract

This enables future compatibility layers.

## Next Planned Step

Container-aware build_bwt:

- accept raw SA32
- OR GLYPHSA1 container

without breaking existing indexes.

## Future Path

raw SA32
    -> GLYPHSA1 entry_width=4
    -> container-aware readers
    -> GLYPHSA1 entry_width=8
    -> SA64 migration
