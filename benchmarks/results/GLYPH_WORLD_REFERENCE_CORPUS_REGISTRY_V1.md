# GLYPH World Reference Corpus Registry V1

- Status: measured identity verification and synthetic hostile gate; research-tier
- Date: 2026-08-27
- Base commit: `da932a11d37fe40a5a3eda0c2133046269e5b62f`
- Registry format: `GLYPH_WORLD_REFERENCE_CORPUS_REGISTRY_V1`

## Purpose

This registry fixes the exact public corpus identities used by the
RLBWT Binary-Safe V2 measurement matrix. A filename alone is not an
identity. Every accepted reference binds:

- a stable reference ID;
- a canonical document name;
- an exact byte length;
- MD5;
- SHA-256;
- a public source;
- whole-file status.

Absolute local paths are not identity-bearing and do not appear in the
registry or committed verification results.

## Default world-reference set

The registry contains seven whole-file references:

1. UCSC hg38 chromosome 1 FASTA;
2. Matt Mahoney enwik8;
3. Matt Mahoney enwik9;
4. Silesia corpus tar;
5. TEXMEX SIFT1M base vectors;
6. TEXMEX GIST1M base vectors;
7. BIGANN DEEP10M base vectors.

Combined verified size:

    9,765,893,325 bytes

## Measured full-file verification

All seven registered files were read sequentially and verified against
their exact byte lengths, MD5 digests and SHA-256 digests.

Measured result:

- registered records: 7;
- verified records: 7;
- verified bytes: 9,765,893,325;
- all files stable during verification: true;
- result path-independent: true;
- full verification result SHA-256:
  `c7330fcb2bc819f38287686fe5b6065d8db3b05cb9f8fe7910d1df545585ec2d`.

The files were read from a shared read-only corpus store. They were not
copied into this repository.

## Synthetic hostile gate

The committed checker executes two positive cases and rejects nineteen
mutation or misuse classes, including:

- malformed, duplicate-key and non-canonical JSON;
- identity-bearing registry changes;
- record removal and record reordering;
- invalid selection and CLI combinations;
- pre-existing output;
- missing, symbolic-link and non-regular reference paths;
- byte-length and digest mismatch.

The checker is deterministic across different working directories and
different requested-reference orders.

Hostile result SHA-256:

    988374cfafd3ec00a6b2fe4e31e20be4c9b65cd02a2efa5ff02477ffd92cfd13

## Boundaries and non-claims

This gate establishes corpus identity and fail-closed verification only.

It does not:

- commit or redistribute corpus payloads;
- measure RLBWT size, build speed or query speed;
- select an RLBWT V2 representation;
- establish a complete sub-1x runtime;
- add this research gate to the top-level proof graph.

## Next measurement gate

Run the binary-safe RLBWT probe on the registered references and report,
for each corpus:

- source bytes;
- BWT row and run counts;
- `r/n`;
- adaptive escape symbol and overhead;
- candidate RLBWT payload bytes and ratio;
- projected rank, locate and metadata budgets.

Full-runtime claims remain forbidden until payload, rank, locate,
metadata and evidence bytes are all measured together.
