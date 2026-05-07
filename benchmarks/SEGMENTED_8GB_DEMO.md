# Segmented 8GB Demo — GLYPH v0.2 branch

Branch:

- feature/segmented-v0.2

Setup:

- 2 logical shards
- each shard uses existing 4GB artifacts
- manifest: manifests/segmented_8gb_demo.json
- runner: glyph_segmented_query_v1.py

Query:

- /tmp/query_41905.bin

Result:

- shard 0 count: 4
- shard 1 count: 4
- total_count: 8

Stability run:

- 10 / 10 successful queries
- total_count remained 8
- query_time_sec: ~0.012–0.013 sec

Interpretation:

This is the first segmented retrieval proof.

It does not yet represent true independent 8GB indexing.
It validates the segmented manifest + multi-shard query + deterministic merge path.
