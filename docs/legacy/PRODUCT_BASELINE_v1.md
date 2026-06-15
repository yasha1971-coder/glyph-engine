# GLYPH PRODUCT BASELINE v1

Status: LOCKED INDEX CORE

Core locked:
- SA32u format
- BWT format
- FM format
- v5 retrieval semantics
- no chunk_map
- chunk_id = SA[i] >> 14

Validated:
- 4GB physical shard
- 8GB logical segmented prototype
- exact retrieval
- deterministic results

Reference implementation:
- glyph_live_retrieve_v5.py
- glyph_segmented_live_v3.py
- parallel shard fan-out
- warm query latency ~1.3-1.7 ms

Not locked:
- HTTP layer
- daemon layer
- router implementation
- API format
- systemd service
- multi-process scaling

Production safety requirements:
- bind only to 127.0.0.1
- one uvicorn worker initially
- request lock around router stdin/stdout
- timeout required
- health endpoint required
- graceful shutdown required
- systemd MemoryMax required

Do not change without new baseline:
- SA format
- BWT format
- FM format
- v5 retrieval semantics
