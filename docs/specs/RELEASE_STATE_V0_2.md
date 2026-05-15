# GLYPH Release State — v0.2

Status:

- experimental
- deterministic retrieval prototype
- segmented retrieval validated

## Validated capabilities

The following paths were validated on real corpora:

- suffix-array construction
- BWT construction
- FM-index construction
- persistent exact retrieval
- segmented shard querying
- deterministic count merging

Validated datasets include:

- HDFS logs
- synthetic mini corpora

## Canonical invariant

GLYPH FM-index v0.x requires:

    indexed_corpus = raw_corpus + appended real 0x00 sentinel

Canonical pipeline:

    raw corpus
    -> prepare_sentinel_corpus_v1.py
    -> build_sa_u32
    -> build_bwt
    -> build_fm

## Current guarantees

GLYPH currently guarantees:

- deterministic exact byte retrieval
- exact suffix-array interval semantics
- deterministic shard merge behavior
- byte-exact counting

## Current limitations

Current v0.x limitations:

- no incremental indexing
- immutable corpora assumption
- no fuzzy matching
- no ranking
- no regex engine
- no semantic search
- no arbitrary 0x00 corpora support

## Segmented retrieval status

Segmented retrieval correctness was validated after fixing sentinel semantics.

Validated result:

- shard-local FM counts match Python ground truth
- merged counts match global corpus truth

## Performance model

GLYPH trades:

- offline indexing cost
- RAM usage
- large index artifacts

for:

- extremely fast repeated exact retrieval

GLYPH is optimized for:

- repeated deterministic queries
- static corpora
- forensic retrieval
- infrastructure-scale exact lookup

## Non-goals

GLYPH is currently NOT intended to be:

- Elasticsearch replacement
- vector database
- semantic retrieval engine
- approximate nearest-neighbor system
- ranking engine

## Stability

Binary formats and APIs may still evolve during v0.x development.

Backward compatibility is not yet guaranteed.
