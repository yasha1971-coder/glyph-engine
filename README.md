# GLYPH

[![ci](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml)

GLYPH is a byte-exact substring retrieval engine over raw bytes.

It is designed for high-speed exact matching without tokenization or scoring.

It is NOT a search engine:
- no ranking
- no fuzzy matching
- no scoring

It performs deterministic exact matches at scale.

---

## ⚡ Try it in 10 seconds

    git clone https://github.com/yasha1971-coder/glyph-engine
    cd glyph-engine
    ./examples/mini/build_mini.sh

Expected output:

    count:    2

This runs a full sentinel-safe pipeline:

- prepares a real appended `0x00` sentinel corpus
- builds suffix array (SA)
- builds BWT
- builds FM-index
- runs a real query

No large datasets required.

## Index your own file

Build an FM index for any small file:

    tools/build_glyph_index_v1.sh /path/to/your/file /tmp/glyph-index

Query an exact pattern:

    ./build/query_fm_v1 /tmp/glyph-index/fm.bin /tmp/glyph-index/bwt.bin "$(printf 'your pattern' | xxd -p -c 999999)"

Important:

- input corpus must not contain `0x00`
- GLYPH v0.x appends a real terminal `0x00` sentinel
- current indexes are optimized for static corpora
- current RAM overhead is high

---

## Build

Before running direct FM queries, build the C++ binaries:

    cmake -S . -B build
    cmake --build build -j2

Required tools:
- CMake
- C++17 compiler
- Python 3
- `xxd`

---

## Documentation

Architecture:
- docs/architecture/ENGINE_OVERVIEW.md

Specifications:
- docs/specs/INDEX_FORMAT_V1.md
- docs/specs/SENTINEL_INVARIANT.md
- docs/specs/KNOWN_LIMITATIONS.md

Benchmarks:
- benchmarks/HDFS_1GB_BENCHMARK.md
- benchmarks/SEGMENTED_FIXED_CORRECTNESS.md

Business / Contact:
- docs/business/CONTACT.md

Project boundaries:
- WHAT_GLYPH_IS_NOT.md

---

## Entry points

| Path | Purpose |
|---|---|
| `examples/mini/` | Start here. Self-contained demo. |
| `tools/build_glyph_index_v1.sh` | Canonical sentinel-safe index builder. |
| `build/query_fm_v1` | Direct FM query binary. |
| `glyph_cli.py` | HTTP client for a running local GLYPH server. |
| `glyph_http_server.py` | Experimental persistent HTTP backend. |
| `glyph_segmented_query_v1.py` | Experimental segmented query path. |

---

## Advanced: HTTP server mode

This mode is experimental and expects prepared index artifacts plus a running local HTTP server.

Note:
- run.sh expects local prepared demo artifacts
- large corpus/index artifacts are not included

Check service:

    curl http://127.0.0.1:18080/health

Query prepared demo data:

    ./glyph_cli.py --hex "$(xxd -p -c 999999 /tmp/query_41905.bin)"

Expected:
- JSON response with exact byte-match shortlist

---

## Core guarantees

- byte-exact substring retrieval
- deterministic results
- no ranking
- no fuzzy matching
- no tokenization
- no semantic interpretation
- sentinel-safe FM-index construction

---

## What problem it solves

Most systems trade accuracy for flexibility:

- grep → scans (slow at scale)
- Elasticsearch → ranks (approximate)
- vector search → approximate similarity

GLYPH does the opposite:

- exact byte matches
- no interpretation
- deterministic results

---

## When to use

- large-scale log search
- binary corpus lookup
- forensic / debugging analysis
- RAG pre-filtering (exact stage before embedding)

---

## Performance

- ~1.3–1.7 ms (warm)
- ~4 ms p99 (4GB shard)
- mmap-based index

RAM note:

Current plain index artifacts are large. The HDFS 1GB benchmark used about
9.4GB RAM for 1GB corpus-scale experiments. This is a known limitation.
Future work must address compressed/sampled SA and better memory economics.

---

## Status

Experimental prototype.

---

## Additional Documentation

Benchmarks:
- benchmarks/HDFS_1GB_BENCHMARK.md
- benchmarks/SEGMENTED_FIXED_CORRECTNESS.md

Security / Legal:
- PATENT_RISK_AUDIT_v2.md

License:

- Apache-2.0


See:
- RUNBOOK_4GB.md
- DEMO_SECURITY.md

---


## Contact

- Website: https://glyph.rs
- Email: contact@glyph.rs

