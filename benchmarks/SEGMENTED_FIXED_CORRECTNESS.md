# Segmented Fixed Correctness — GLYPH v0.2

Branch:

- feature/segmented-v0.2

Purpose:

Validate segmented retrieval after fixing FM sentinel semantics.

Root cause fixed:

- FM-index requires corpus + real appended 0x00 sentinel.
- Previous builds used synthetic BWT sentinel without appending a real sentinel to the indexed corpus.
- This caused shifted FM intervals and undercounting.

Fixed build flow:

    raw corpus
    -> tools/prepare_sentinel_corpus_v1.py
    -> build_sa_u32
    -> build_bwt
    -> build_fm

Dataset:

- HDFS 1GB
- split into two independent 512MB shards
- each shard indexed through tools/build_glyph_index_v1.sh

Manifest:

- manifests/segmented_1gb_fixed.json

Pattern:

- blk_-1000095285706020638

Ground truth:

- shard0 Python bytes.count: 17
- shard1 Python bytes.count: 10
- total Python bytes.count: 27

GLYPH segmented result:

- shard0 FM count: 17
- shard1 FM count: 10
- merged total_count: 27

Conclusion:

Segmented retrieval correctness is restored when shards are built through the sentinel-safe indexing pipeline.
