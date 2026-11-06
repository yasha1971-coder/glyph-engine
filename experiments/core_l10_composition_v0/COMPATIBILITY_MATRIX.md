# CORE-L10 · Gate 0 — Composition Reference / RLB2–EF compatibility matrix

Date: 2026-11-06. Status: **SOURCE REVIEWED; HISTORICAL CI REPLAY REQUIRED.**
Source commits, not aliases:

- Historical Composition Reference V1: `81628f843dc8aa633c65594a66d2386bfc3cb5c2`.
- Frozen binary-safe RLB2/RLR2: `c33cb493bc75f13ee4a904fabb190a29a196c94b`.
- CORE-L9-S prerequisite: `e7f0e3b16381519c8b0269f6fea0ff11feb0cc80` (Draft PR #27).
- Issue #28 is the existing task tracker. No subsequent issue is required.

## Grounded contract comparison

| Topic | Composition Reference V1 (historical source) | L9-S actual substrate | Decision |
|---|---|---|---|
| Physical block | Independently verified **GLYPH_OPERATOR_RUNTIME_INDEX_V1**, with one or more documents, source snapshots, complete marker, source manifest and physical runtime commitment | A pinned **RLB2/RLR2/GL3EF001** single-source search index; no operator V1 manifest or canonical document mapping | **CONTRACT_MISMATCH**: never label an RLB2 leaf as a compliant V1 runtime block |
| Global coordinates | `(global_doc_id, doc_offset)`, sorted lexicographically by IDs and offsets | `corpus_byte_offset`, single corpus; no doc mapping | **ADAPTER_REQUIRED**: one indexed leaf per explicitly declared nonempty document |
| Empty documents | Valid as one member of a **nonempty** runtime block; contribute zero matches and still consume one doc ID | Physical RLB2 does not yet have a proved normatively equivalent empty-doc operator package | **RESEARCH EXCEPTION**: explicit hashed empty-doc record, independently verified, not a normative indexed block |
| Corpus identity | SHA256 of `GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\x00 || U64BE(doc count) || Σ(U64BE(doc_id)||U64BE(length)||SHA256(doc))` | L9 single source root SHA and corpus byte identity | **SAFE TO RECOMPUTE** exactly from ordered document records; do not alias an L9 root to a Composition root |
| Source manifest identity | SHA256 of `GLYPH_OPERATOR_CORPUS_MANIFEST_V1\x00` plus canonical ordered path bytes and document commitments | L9 root lacks original multi-doc path/ID mapping | **SAFE TO RECOMPUTE** only from a separately committed canonical path sequence with independent validation |
| Composition root identity | SHA256 over actual **V1 operator** block `runtime_index_id`, manifests and layout | No V1 operator runtime index IDs for RLB2 leaves | **CONTRACT_MISMATCH**: use distinct `research_layout_id` with `GLYPH_CORE_L10_LAYOUT_V0` domain separation. **Never** output a fabricated `composition_root_id` |
| Coverage | Exact full ordered block identity: `expected == verified == queried` | Single-corpus `PROVEN_EMPTY` only after full authoritative source scan | **TRANSFERABLE SEMANTIC RULE**: verify all declared RLB2 leaf identities, query all leaves before global `PROVEN_EMPTY`; absent/mutated/stale leaf = `UNTRUSTED` |
| Bounded locate | Global sorted prefix `max_offsets`, count unaffected | RankV2 bounded FM rows may not be sorted text offsets; L9 full results sorted | **ADAPTER REQUIRED**: calculate each leaf's **complete** offsets first on small fixture, globally sort and cap; never cap each leaf independently |
| Boundary policy | `DOCUMENT_LOCAL_MATCHES_ONLY_V1`: no cross-document or cross-block match | Each physical RLB2 leaf is one source byte sequence | **SAFE IN SINGLE-DOC LEAF PROFILE** if empty docs are explicit and query occurs only within one leaf |
| Repartition | Same ordered documents, same logical `runtime_corpus_id`, different `composition_root_id` for valid V1 operator layouts | Grouping leaf references does not rebuild physical V1 blocks | **RESEARCH SEMANTIC TEST ONLY**: different `research_layout_id`, same logical ordered corpus identity and doc coordinates; not a V1 layout root |
| Append-only | Existing source identity is ordered by canonical paths and IDs | L9-S has no append-time semantics | **REQUIRES POLICY**: keep prior document order unchanged with *monotonically increasing raw canonical paths*; out-of-order path appends must reject, not silently renumber IDs |
| Proof authority | V1 root and operator evidence bundle are reference-stage artifacts; user trust/currentness separate | External caller-supplied SHA-256, offline byte scan, no signing/currentness | **NOT EQUIVALENT**; claim only pinned-byte research truth, not real-world freshness or O3/O4 |
| Full source replay | Old operator index and independently verified source snapshots | L8 native RLB2 full restore and L9-S transient SHA-verified source for independent oracle | **PROTOTYPE**: leaf source identity may be verified with independent native RLB2 inverse, but source must not be retained as a second archive |

### Important append-id collision

O1 sorts raw source path bytes before assigning IDs. If an arbitrary newly ingested filename compares *before* existing paths, recomputing the historical source manifest changes old global IDs. Therefore **general-purpose arbitrary-path append with stable document IDs is NOT provided by V1**. For this research gate, allocate monotonic canonical internal names (e.g. `vault/00000000.bin` through `vault/00000011.bin`) while treating original user filename as separately recorded metadata rather than assuming O1's path identity remains unchanged. Production naming/mapping requires separate owner-approved semantics.

### Exact compatibility boundary

The new adapter may use established identities `runtime_corpus_id` and `source_manifest_id` **only if it reproduces their exact normative preimages**. It must introduce separate, explicitly nonnormative `research_layout_id` and `research_snapshot_id`. These do not become composition_root_id or runtime_index_id by renaming them. A future full Composition V1 adapter must construct and verify genuine V1 operator runtime units (or obtain an owner-approved new normative version) before claiming operator-level compatibility.

### Gate 0 independent executable check

The reference CI must run the **untouched** `tools/check_composition_reference_v1.py` from the pinned historic checkout, which itself invokes `replay_composition_reference_v1.py`. The source-declared matrix has M01–M25 all `EXACT`, but CI **must independently confirm** `normative_exact_count=25`, `normative_open_count=0` and `GLYPH COMPOSITION REFERENCE OK`. Historical executable PASS is not proof that L9's physical operator artifacts match V1.

### Next implementation scope

A separate research file under this directory may build >=9 real single-document RLB2/RLR2/EF leaves; hash every leaf and an explicit empty document; verify two distinct *research groupings* of an identical ordered document set; append >=3 immutable leaves; and replay exact source bytes, entire coverage and globally canonical coordinates using an independent program. Until its actual CI passes, those are **tests to perform, not GLYPH capabilities**.
