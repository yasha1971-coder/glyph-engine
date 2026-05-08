# GLYPH Index Format v1

Status:

- experimental
- unstable during v0.x

Purpose:

Define the binary layout and invariants of GLYPH FM-index artifacts.

## FM binary header

Magic:

    FMBINv1\0

Layout:

    offset  size  field
    --------------------------------
    0       8     magic
    8       8     corpus_bytes (uint64)
    16      4     checkpoint_step (uint32)
    20      8     num_blocks (uint64)
    28      2048  C[256] uint64 table
    ...           checkpoints

Checkpoint layout:

    checkpoints[num_blocks][256]

Stored as:

    uint32 per symbol count

Meaning:

Each checkpoint stores cumulative occurrence counts
for all byte values before the corresponding block.

## BWT assumptions

GLYPH v0.x assumes:

    indexed_corpus = raw_corpus + appended 0x00 sentinel

Required invariant:

- raw corpus must not contain 0x00
- appended sentinel must be unique

Failure to satisfy this invariant may produce:

- shifted FM intervals
- incorrect occurrence counts
- deterministic undercounting

## Query semantics

GLYPH performs exact byte matching.

Returned result:

    suffix-array interval [l, r)

Match count:

    r - l

## Segmented manifests

Segmented retrieval uses a manifest describing shards.

Each shard contains:

- corpus
- suffix array
- BWT
- FM index

Global retrieval result is produced by:

- independent shard query
- deterministic count merge

## Compatibility

v0.x formats are not yet stable.

No backward compatibility guarantees currently exist.
