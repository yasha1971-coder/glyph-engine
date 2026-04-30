# GLYPH 8GB SEGMENTED PROTOTYPE PLAN

Goal:
Validate segmented retrieval architecture before building larger indexes.

Prototype:
- shard0 = existing 4GB index
- shard1 = same existing 4GB index reused logically
- global_chunk_id = shard_id * chunks_per_shard + local_chunk_id

Why:
- tests fan-out
- tests merge
- tests result normalization
- avoids rebuilding 8GB immediately

Required:
- shard config
- segmented retriever
- merged JSON output

Next implementation:
- glyph_segmented_retrieve_v1.py
