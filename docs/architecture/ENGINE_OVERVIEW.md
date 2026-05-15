# GLYPH Engine Overview

GLYPH is a deterministic byte-exact retrieval engine for static corpora.

It is not a search relevance engine, vector database, fuzzy matcher, or regex engine.

Core idea:

- build an offline FM-index over a prepared corpus
- answer repeated exact byte queries without rescanning the corpus
- preserve deterministic retrieval semantics

Current architecture:

- raw corpus
- sentinel-safe prepared corpus
- suffix array
- BWT
- FM-index
- persistent query backend
- segmented manifest layer

Canonical build invariant:

GLYPH FM-index v0.x must index:

    corpus + real appended 0x00 sentinel

The canonical build flow is:

    raw corpus
    -> prepare_sentinel_corpus_v1.py
    -> build_sa_u32
    -> build_bwt
    -> build_fm

Segmented retrieval:

GLYPH v0.2 introduces shard manifests.
Each shard has an independent corpus, SA, BWT, and FM index.
Queries are dispatched across shards and merged deterministically.

Known v0.x limitation:

The current sentinel-safe mode requires input corpora without 0x00 bytes.
Arbitrary raw bytes require a future 257-symbol alphabet or out-of-band sentinel representation.

Correctness status:

The HDFS undercount bug was traced to missing real appended sentinel semantics.
After sentinel-safe indexing, segmented retrieval matches Python byte-count ground truth.
