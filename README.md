# GLYPH

## What is GLYPH?

GLYPH is a pre-indexed exact-byte retrieval and evidence layer for fixed corpora.

It is useful after a corpus becomes an archive or evidence object, when the goal is not just to find a string, but to show exactly where it was found and how another person can reproduce the result.

GLYPH produces a reproducible exact-byte evidence chain:

- corpus hash
- index manifest hash
- query hash
- FM interval
- match count
- exact offsets when the locate layer is available
- byte-checks
- replay command
- Audit Artifact V0
- Evidence Case V1

GLYPH is not a cold-start grep replacement, SIEM replacement, ELK/Splunk replacement, semantic search engine, legal proof system, or zero-knowledge proof system.

If you cloned this repository, start with:

    ./verify.sh

Then see:

- examples/mini/
- examples/public-evidence-demo/
- examples/public-evidence-demo/run_pizza_50mb_demo.sh
- docs/specs/

---

[![CI status](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml)

## What GLYPH verifies

GLYPH is useful when a corpus is already fixed, large, remote, or expensive to rescan.

The core workflow is:

    fixed corpus
    → corpus hash
    → exact query bytes
    → match count
    → offsets
    → byte-checks
    → replayable audit artifact

The verifier does not need to search the whole corpus again.

If they have the same corpus, they can seek directly to the recorded offsets, read the bytes, and verify that the exact query is present there.

This is the narrow claim:

    exact bytes existed at exact offsets in this exact corpus state.

This is not a claim of semantic truth, attribution, intent, legal proof, or full incident reconstruction.

Start small:

    ./verify.sh

Mini evidence flow:

    ./examples/mini/run_mini.sh

Public-style 50MB corpus demo:

    ./examples/public-evidence-demo/run_pizza_50mb_demo.sh

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

verify and reproduce exact byte presence inside a fixed static corpus.

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

## Who is this for?

GLYPH is looking for people who need to verify exact bytes in a fixed corpus and let someone else replay the result.

This may matter for:

- DFIR and incident response
- malware or binary corpus analysis
- fixed log and audit corpora
- legal, compliance, and review workflows
- RAG / LLM citation grounding
- succinct data structure research

If this is a real pain for you, tell us your use case:

https://github.com/yasha1971-coder/glyph-engine/issues/3

Or join the discussion:

https://github.com/yasha1971-coder/glyph-engine/discussions/4

If you know a tool that already provides the same replayable exact-byte evidence chain, please tell us that too.

## Experimental compressed bounded evidence path

GLYPH also has an experimental compressed bounded evidence path over RLBWT runtime.

This path is intended for review and engineering validation.

It demonstrates:

* compressed BWT runtime using bwt.rlbwt
* rank support using bwt.rlbwt.rank
* exact FM interval and match count
* bounded offset recovery using max_offsets
* byte verification for returned offsets
* replay verification of the bounded evidence artifact

The current verified artifact format is documented in:

- `docs/specs/RLBWT_BOUNDED_EVIDENCE_SPEC_V1.md`
- `docs/review/RLBWT_BOUNDED_EVIDENCE_REVIEW_PATH_V1.md`
- `docs/review/GLYPH_CURRENT_TECHNICAL_STATE_V1.md`

The tiny reproducible fixture is:

./tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh

The fixture is included in the top-level verifier:

./verify.sh

Current verified tiny fixture result:

- query: `the`
- FM interval: `[65, 68]`
- match count: `3`
- max offsets: `2`
- returned offsets: `[43, 55]`
- byte check: PASS
- replay verifier: PASS

Important non-claims:

* This is not exhaustive locate when bounded=true.
* This is not semantic search, ranking, fuzzy matching, or token search.
* This is not a legal proof by itself.
* This is an experimental compressed exact-retrieval evidence path with reproducible bounded offsets.

For high-count queries, bounded evidence separates exact counting from exhaustive offset enumeration:

full exact count
+ deterministic bounded offsets
+ byte verification
+ replay verification

## Start Here: Verified Path

GLYPH is a verifiable exact-byte retrieval and evidence engine for committed byte corpora.

The fastest way to verify the current repo state is:

```bash
git clone https://github.com/yasha1971-coder/glyph-engine.git
cd glyph-engine
./verify.sh
```

Expected verifier lines include:

```text
[verify] RLBWT bounded evidence tiny fixture
[verify] Structural Fingerprint V0 replay smoke
[verify] structural fingerprint replay ok
VERIFY OK
```

Current verified capabilities:

- replayable exact-byte retrieval evidence on a bounded RLBWT fixture
- portable evidence bundle replay
- deterministic Structural Fingerprint V0 artifact
- Structural Fingerprint replay verification from source bytes

Important non-claims:

- GLYPH is not a codec predictor
- GLYPH is not a compression optimizer
- GLYPH is not a machine-learning classifier
- GLYPH is not a legal-proof oracle

If this solves or nearly solves a real use case, open an issue or discussion:

- Issue: https://github.com/yasha1971-coder/glyph-engine/issues/3
- Discussion: https://github.com/yasha1971-coder/glyph-engine/discussions/4

## Structural Fingerprint Replay

GLYPH also includes a deterministic structural measurement artifact:

```text
GLYPH_STRUCTURAL_FINGERPRINT_V0
```

This is not a codec predictor and not a compression optimizer.

It is a replayable structural fingerprint of a byte corpus, including:

- source byte identity
- SHA256
- byte statistics
- entropy profile
- anchor repeat profile
- optional BWT run profile
- explicit non-claims

The artifact can be regenerated and replay-verified from the original source bytes:

```bash
python3 tools/glyph_structural_fingerprint_v0.py examples/mini/out/corpus.bin --out /tmp/mini_structural_fingerprint_v0.json
python3 tools/replay_structural_fingerprint_v0.py /tmp/mini_structural_fingerprint_v0.json --out /tmp/mini_structural_fingerprint_replay_v0.json
```

The top-level verifier includes a smoke test:

```bash
./verify.sh
```

Expected verifier lines include:

```text
[verify] Structural Fingerprint V0 replay smoke
[verify] structural fingerprint replay ok
VERIFY OK
```

This extends GLYPH from replayable exact-byte retrieval evidence into replayable structural corpus measurement.
