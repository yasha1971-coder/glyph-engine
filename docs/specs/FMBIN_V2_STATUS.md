# FMBINv2 Status

Status: implemented

## Summary

FMBINv2 adds internal checkpoint payload verification to GLYPH FM artifacts.

Before FMBINv2:

- FM artifact had magic/version marker
- checkpoint payload corruption could pass header checks

After FMBINv2:

- FM artifact uses magic `FMBINv2\0`
- checkpoint payload byte size is stored
- checkpoint payload FNV-1a 64-bit checksum is stored
- FM readers verify payload size and checksum before query

## Updated components

- `src/build_fm.cpp`
- `src/query_fm_v1.cpp`
- `src/query_fm_server_v1.cpp`
- `src/query_fm_batch_v1.cpp`

## Manual corruption test

Procedure:

    cp examples/mini/out/fm.bin /tmp/fm_corrupt.bin
    printf '\xFF' | dd of=/tmp/fm_corrupt.bin bs=1 seek=4096 count=1 conv=notrunc
    ./build/query_fm_v1 /tmp/fm_corrupt.bin examples/mini/out/bwt.bin 6572726f72

Expected result:

    ERROR: fm checkpoint checksum mismatch

Observed result:

    ERROR: fm checkpoint checksum mismatch

## Meaning

FMBINv2 converts FM checkpoint corruption from silent risk into fail-fast
artifact rejection.

This strengthens Tier 3 strict verification prerequisites.
