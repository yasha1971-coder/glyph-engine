GLYPH RETRIEVAL CONTRACT V1

Purpose

Define GLYPH as a versioned deterministic retrieval contract.

A valid GLYPH runtime is not only code.
It is a compatible tuple of:

corpus
manifest
FM artifact
golden query fixtures
query protocol
persistent server behavior
HTTP bridge behavior

Contract tuple

corpus_hash
manifest_version
fm_artifact_version
fm_payload_checksum
golden_fixture_hash
query_protocol_version
server_protocol_version
http_protocol_version

Current versions

manifest_version: GLYPH_INDEX_MANIFEST_V1
fm_artifact_version: FMBINv2
query_protocol_version: GLYPH_QUERY_PROTOCOL_V1
batch_protocol_version: GLYPH_QUERY_BATCH_PROTOCOL_V1
http_protocol_version: GLYPH_HTTP_QUERY_V1

Rule

All parts move together.

If one part changes without the others being regenerated or revalidated,
the system must fail hard or mark the state incompatible.

Silent drift is worse than hard failure.

Determinism invariant

Identical corpus
+ identical artifacts
+ identical protocol versions
+ identical query input

must produce identical JSON output.

Current implemented guarantees

- sentinel-safe corpus invariant
- FMBINv2 magic/version validation
- FM payload size validation
- FM payload checksum validation
- manifest integrity checks
- golden query fixtures
- CLI JSON query protocol
- persistent server JSON protocol
- HTTP query bridge
- CI regression coverage

Planned extensions

- batch query protocol implementation
- shard-aware protocol
- corpus-level full hash contract
- server capability fingerprint
- runtime environment fingerprint
- SA container integration
- SA64 compatibility
