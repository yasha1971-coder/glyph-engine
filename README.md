# GLYPH

[![ci](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml)

## Verify GLYPH in one command

```bash
git clone https://github.com/yasha1971-coder/glyph-engine.git
cd glyph-engine
./verify.sh
```

Expected output:

```text
VERIFY OK
```

This checks the canonical mini verification pipeline.

See:

- docs/specs/GLYPH_VERIFICATION_CLAIM_V1.md

---

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

Roadmap:
- docs/roadmap/PERSISTENT_RETRIEVAL_ROADMAP_V1.md

Benchmarks:
- benchmarks/HDFS_1GB_BENCHMARK.md
- benchmarks/SEGMENTED_FIXED_CORRECTNESS.md

## Public reproducible benchmark

GLYPH now includes an enwik9 public benchmark runbook.

See:

- benchmarks/ENWIK9_PUBLIC_BENCH_RUNBOOK_V1.md
- benchmarks/LAYOUT_PROFILE_MATRIX_V1.md
- benchmarks/PUBLIC_BENCH_EXPECTATIONS_V1.md

The enwik9 corpus was verified to contain 0 null bytes, making it compatible with the current sentinel-safe GLYPH v0.x pipeline.

Current public benchmark focus:

- SA build
- BWT build
- FM layout scaling
- checkpoint_step memory economics
- cold CLI profile

Persistent raw FM latency is intentionally not claimed yet.

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

GLYPH is built for one narrow class of problems:

prove or reproduce exact byte presence inside a static corpus.

The current target is not general search.

The strongest fit is:

* static archives
* frozen datasets
* provenance investigations
* training-data membership checks
* exact evidence over immutable corpora

GLYPH is useful when the question is:

```text
Did these exact bytes exist in this exact corpus state?
```

Not:

```text
What is semantically relevant?
```

---

## When to use

Use GLYPH when:

* the corpus is static or versioned
* exact byte identity matters
* repeated exact queries are expected
* reproducibility is more important than ranking

Do not use GLYPH as:

* a general search engine
* a live log analytics system
* an embedding/vector database
* a replacement for grep on one-off scans

---

## Performance

Performance numbers are experimental and depend heavily on corpus size,
index layout, cache state, and runtime mode.

Current public focus:

* correctness
* reproducibility
* deterministic verification
* memory-model transparency

Known limitation:

Current plain index artifacts are large. The HDFS 1GB benchmark used about
9.4GB RAM for 1GB corpus-scale experiments. This is a known limitation.

Future work must address compressed/sampled SA and better memory economics.

---

## Status

Experimental prototype.

---

See:
- RUNBOOK_4GB.md
- DEMO_SECURITY.md

---

## Contact

- Website: https://glyph.rs
- Email: contact@glyph.rs

