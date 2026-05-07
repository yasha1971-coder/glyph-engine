# Segmented Real 1GB Retrieval — GLYPH v0.2

Branch:

- feature/segmented-v0.2

Setup:

- HDFS 1GB corpus
- split into 2 independent 512MB shards
- independent SA/BWT/FM built per shard

Manifest:

- manifests/segmented_1gb_real.json

Query:

- blk_-100000266894974466

Result:

- shard0 count: 18
- shard1 count: 0
- merged total_count: 18

Latency:

- query_time_sec: ~0.000319

Interpretation:

This is the first real segmented retrieval validation.

Each shard has an independent corpus and independent FM index.

The segmented query layer:
- dispatches queries across shards
- merges deterministic counts
- preserves exact retrieval semantics
