# RLBWT_CONTAINER_V1_ROUNDTRIP

Status: measured local roundtrip test  
Date: 2026-06-26

## Purpose

Validate minimal RLBWT Container V1 encode/decode roundtrip over existing GLYPH `bwt.bin` artifacts.

This does not implement rank/select yet.

It proves container encoding and decoding are bit-perfect.

## Format

Container magic: `RLB1`

Payload:

    (symbol:uint8, run_length:ULEB128) repeated for every BWT run

Header stores:

- version
- original BWT length
- run count
- raw BWT SHA256

## Results

| label | source_bytes | encoded_bytes | ratio_vs_bwt | sha256_match |
|---|---:|---:|---:|---|
| pizza50 | 50,000,001 | 31,922,909 | 0.638x | true |
| xz_cve | 38,928 | 12,790 | 0.329x | true |
| synthetic_logs50 | 50,000,001 | 7,404,269 | 0.148x | true |

## SHA256 verification

Pizza 50MB:

- source_sha256: `eca058f8d67a98cb665fe36b24bb8a1195504f32636171ae2625a2b251cddd06`
- decoded_sha256: `eca058f8d67a98cb665fe36b24bb8a1195504f32636171ae2625a2b251cddd06`

XZ CVE corpus:

- source_sha256: `bbd989643d6b673bf7ac965c52b2697edc8491902848f2a04dfbc6ad4a1f7190`
- decoded_sha256: `bbd989643d6b673bf7ac965c52b2697edc8491902848f2a04dfbc6ad4a1f7190`

Synthetic logs 50MB:

- source_sha256: `62c13795444ba83ef7c988a40b8668bc30795436a2b0efe24aab5d48b0825325`
- decoded_sha256: `62c13795444ba83ef7c988a40b8668bc30795436a2b0efe24aab5d48b0825325`

## Interpretation

RLBWT Container V1 gives a bit-perfect compressed representation of raw `bwt.bin`.

This confirms that raw BWT storage can be reduced substantially before implementing a full r-index.

Measured compression:

- normal text: 0.638x of raw BWT
- small real-event text corpus: 0.329x of raw BWT
- repetitive synthetic logs: 0.148x of raw BWT

## Current boundary

This is not yet a compressed-BWT query engine.

Missing pieces:

- rank over RLBWT
- LF mapping over compressed BWT
- locate compatibility
- query correctness tests
- query latency tests

## Next step

Design RLBWT rank blocks.

Minimal next target:

    rank(c, pos) over RLBWT without fully decoding bwt.bin

Once rank works, GLYPH can attempt query over compressed BWT.
