# Binary-safe RLBWT Runtime V2 on enwik8

## Status

This report records an experimental, measured binary-safe exact substring
runtime for the canonical 100,000,000-byte enwik8 corpus.

The result is a combined lossless storage and exact count/locate index. It is
not presented as a general-purpose compression-codec replacement or as an
optimized production latency result.

## Canonical corpus

- reference: `matt-mahoney-enwik8`
- source bytes: `100000000`
- MD5: `a1fa5ffddb56f4953e226637dabbb36a`
- SHA-256:
  `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`
- BWT rows including the logical sentinel: `100000001`

The corpus payload, canonical SA, and canonical BWT are not committed.

## Measured runtime size

| Runtime component | Bytes | SHA-256 |
|---|---:|---|
| RLB2 run container | 73,365,105 | `4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3` |
| RLR2 rank checkpoints | 12,844,028 | `e76a49db554860bf907aa307a503240d1e71ef0400282c34548a9726a3f22719` |
| LOC1 samples, step 128 | 12,500,040 | `1ba04acce3b2ebea10ae3ab703138c1a33e7c9caa27fe7a1056d4ed8c44f7f6f` |
| Canonical runtime manifest | 828 | `ddcaebdf9b518f63be4d4a47a801b89d155a5372ebeb6ada8c6866dcd268b827` |
| **Packaged runtime total** | **98,710,001** | — |

Measured packaged ratio:

`98710001 / 100000000 = 0.987100010`

Measured margin below the source size:

`100000000 - 98710001 = 1289999 bytes`

The construction-only 400,000,064-byte SA and 200,000,058-byte canonical BWT
are excluded from runtime size because neither is required by packaged query
execution.

## Structure

- RLB2 rows: `100000001`
- RLB2 runs: `36661040`
- run ratio `r/n`: `0.366610400`
- RLB2 payload bytes: `73364945`
- RLR2 rank step: `8192`
- RLR2 checkpoints: `12209`
- RLR2 counter width: `32` bits
- RLR2 record bytes: `1052`
- locate sample step: `128`
- locate records: `781251`

## Correctness evidence

- RLB2 decode is bit-identical to the canonical binary-safe BWT.
- All `12209 × 257 = 3137713` RLR2 counters were independently compared
  against a direct scan of all `100000001` canonical BWT symbols.
- All `781251` physical locate records were compared with canonical SA rows.
- The enwik8 query matrix proved exact counts and offsets against direct corpus
  search.
- The matrix included `19` count patterns, `27` locate patterns, and `24`
  forced non-sampled rows.
- Forced non-sampled rows executed `2645` LF steps; the observed maximum was
  `391`.
- The portable hostile gate verified `35` positive cases and rejected all
  `40` mutations.
- Hostile-gate output was identical from different working directories.
- Manifest-bound execution verifies exact filenames, sizes, formats, hashes,
  corpus identity, rank step, sample step, row count, and runtime byte total.

## Prototype resource measurements

These measurements describe the current Python research implementation, not
an optimized native runtime.

| Operation | Wall time | Peak RSS |
|---|---:|---:|
| canonical SA build | 6.15 s | 1,243.125 MiB |
| canonical BWT build | 3.45 s | 671.078 MiB |
| RLB2 encode | 85.33 s | 56.836 MiB |
| RLB2 inspect | 4.93 s | 40.324 MiB |
| RLB2 decode | 68.04 s | 34.598 MiB |
| RLR2 build | 60.09 s | 40.086 MiB |
| RLR2 inspect | 1.47 s | 28.391 MiB |
| LOC1 construction and complete record verification | 0.72 s | 423.852 MiB |

Measured query matrix:

- count median: `39.998 ms`
- count p95: `73.748 ms`
- locate median: `665.833 ms`
- locate p95: `1534.187 ms`
- runtime startup: `43.338 ms`

These latency values must not be generalized to a future C++ or
memory-mapped implementation.

## Claim boundary

Proven for the exact artifacts and hashes in this report:

1. The packaged runtime is lossless and binary-safe.
2. It provides exact substring count and locate.
3. Its complete measured packaged size is below the enwik8 source size.
4. Its three runtime artifacts are bound by a canonical fail-closed manifest.
5. Rank, locate, query, and hostile-gate evidence close the experimental
   runtime claim for enwik8.

Not claimed:

- universal sub-1x size on arbitrary corpora;
- competitive optimized latency;
- replacement of general compression codecs or database systems;
- inclusion of construction workspace in the runtime footprint.
