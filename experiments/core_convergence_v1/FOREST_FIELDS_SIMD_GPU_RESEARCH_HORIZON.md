# GLYPH CORE — Index Forest, Field Masks, SIMD and GPU/VRAM Research Horizon

Record date: 2026-11-05
Status: **RESEARCH HYPOTHESES / PRECONDITION-FIRST** — no released capability or competitive superiority claimed.
Owner sources: technical-scientific passport v0.1, 2026-08-27; subsequent owner architecture proposal and GLYPH historical evidence.
Purpose: prevent GLYPH's original proof-carrying exact-search+compression goal from drifting into either a conventional RAG stack or a disconnected GPU accelerator.

## 0. The mandatory shared contract

GLYPH's existing *P1–P12 → R0–R6 → O1–O6 → I0* exact, fail-closed and independently replayable proof chain is the semantic reference. Frozen Laptop/OVH GLYPH V1 has separately proven a human-gated authenticated historical-query path. Both are **required inheritance**, not optional post-processing.

A future physical implementation may have multiple independently immutable blocks, a separate compressed payload, field bitmaps and GPU working sets. They MUST all bind to the **same ordered logical corpus/generation**, declare coverage, produce canonical \`(doc_id, doc_offset)\`, and permit independent verification and exact byte restoration. No query-heuristic, embedding, model, cache, GPU result flag or unsigned sidecar may turn UNKNOWN into PROVEN_EMPTY.

Keep composition identities separate:
- \`composed_corpus_id\`: partition-independent ordered document bytes and identity;
- \`composition_root_id\`: partition/layout-dependent manifest identity.
A repartition may change the latter while preserving the former and canonical matches.

The existing Composition Reference's 25 EXACT mutations are **research-tier executable semantics**. Do not implement runtime Index Forest until its outstanding evidence/acceptance preconditions and required graph integration are resolved.

## 1. Index Forest / multi-block exact lookup (HYPOTHESIS)

Candidate representation: append-only immutable block manifests carrying document-order spans, schema/generation IDs, compressed self-index/locate identities and verified coverage summaries, composed under a root. Shard by **whole canonical documents** first; do not silently allow document-crossing matches. Provenanced \`block_id\` is not a global document ID.

**Count** of a pattern = sum of verified per-block counts over all applicable documents; **locate** = stable canonical merge of all verified document-local offsets. Block skipping is permitted only when an independently checked conservative synopsis can prove no hit in that block; never let probabilistic Bloom false positives cause an omitted block. A Bloom filter has no false negatives under its declared correct construction, but its *integrity, parameterization and complete binding to corpus versions* are proof obligations.

Required negative tests: omitted/duplicated block; reordered/duplicated/empty documents; same document under different block partition; present cross-physical-block match inside a logical document (if the future partition scheme splits documents); forbidden cross-document matches; stale child/root binding; incomplete query coverage; partial index unavailable; fork/rollback/root authority. Incomplete coverage => UNKNOWN/UNTRUSTED, not PROVEN_EMPTY.

**Falsifier:** a count/offset difference vs direct byte oracle, a false PROVEN_EMPTY, unstable \`composed_corpus_id\` under repartition, or silent loss of source bytes.

## 2. Field-aware / attribute routing (HYPOTHESIS)

Keep metadata/field schema in a versioned authenticated sidecar or compact binary table bound to the root, rather than embedding mutable high-level policy in BWT bytes. A minimal immutable record may hold \`doc_id\`, \`schema_id\`, \`field_id\`, \`field_value/type\`, \`generation\`, \`field_mask_root\`. Use canonical types/encoding; distinguish metadata values from byte substrings and true document body content.

Exact candidate positions can be intersected with an authenticated doc-ID bitmap for a field, selected source version, or access policy. A block-level field synopsis may reduce I/O when it is version-bound and false-negative-free. This gives **correctly scoped field filtering**, not SQL joins/aggregations, free ACL or vector similarity.

Watch for ACL-sensitive existence leaks: whether an unauthorized document exists can itself be sensitive. The API should return the policy-authorized result set without leaking forbidden counts; version and authorization identity must be explicit. Private data cannot be trusted to a GPU cache or replica lacking corresponding access controls.

**Falsifier:** wrong-field match, unbound/mutable metadata, cross-generation leakage, falsely certified absence because a field mask was stale, or access-control bypass.

## 3. SIMD/AVX rank and locate (HYPOTHESIS)

Keep existing RLBWT/RLB3X formats unchanged initially. Add benchmark-only adapters using AVX2/AVX-512/POPCNT where useful for rank/select, block decode, scanning, varints and bitmap intersections. Retain scalar reference fallback and explicit CPU capability dispatch. Focus on cost of *random rank access, startup scan, LF work and cold vs warm cache*; vectorization cannot fix poor memory locality or guarantee bounded LF hops.

GLYPH's historically observed LOC1 sample step 128 did NOT guarantee LF<=127; a max of 391 was recorded on enwik8. A separate sampling invariant is needed for a strict locate bound. Do not confuse a faster LF step with a bound on the number of steps.

**Falsifier:** any bitwise count/locate mismatch, data-dependent timing cliffs beyond declared SLA, no statistically reproducible improvement after reporting index overhead and memory, or a changed frozen format without explicit contract versioning.

## 4. Bowtie/BWA lessons without sacrificing exactness

Bowtie 2 uses FM-indexed multi-seed read alignment but can miss otherwise valid approximate alignments due to heuristics. This is NOT acceptable for a GLYPH API promising exhaustive exact matches.

BWA-MEM2 offers useful engineering lessons in SA/index compression, SIMD dispatch and batch throughput; its own human-genome FM/read-mapping system reported ~10 GB index footprint. Bowtie 2 manual cites ~3.2 GB RAM for its human-genome FM-index. These measurements use different alignment functions, genome representations, included metadata and hardware. Neither proves GLYPH's proposed 2.1 GB full indexed human-genome profile.

For 3 billion DNA bases, ideal 2-bit ACGT payload is approximately 0.75 GB decimal **before** ambiguity symbols, rank/locate, index navigation, manifests, proof and recovery. 2.1 GB remains a conditional measurable research target, not a derived guaranteed size. O(m) count presumes efficient rank; reporting millions of individual locate results/output is at least proportional to the number of returned hits. No universal "0% decompression": a compact rank path may decode partial blocks even when no complete corpus extraction occurs.

See https://bowtie-bio.sourceforge.net/bowtie2/manual.shtml and https://github.com/bwa-mem2/bwa-mem2 .

## 5. What GLYPH may replace, and what it cannot claim

**Potential narrow replacement:** exact substring/byte presence verification, authenticated immutable evidence retrieval, static snapshot string locate, and some redundant per-source provenance mapping/caching when latency and memory tests support it.

**Not implied:** full Elasticsearch relevance analysis + hybrid ranking + aggregations + distributed CRUD, Redis's data structures/session/cache use cases, Snowflake's SQL/storage/transactions/compute, or vector DB ANN/KNN semantics. Existing Elasticsearch and Snowflake Cortex Search already support lexical/vector combinations and structured filters; "nobody has this combination" is false if stated without GLYPH's *specific proof contract*. A comparative matrix must define exact semantics and complete storage/verify costs. Unsupported AWS cost/traffic/cluster numbers and universal 2–3x search-index ratios are NOT to be used in public claims.

Reference docs:
- https://www.elastic.co/docs/solutions/search/vector
- https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview

For probabilistic AI interpretation: ANN/search can **propose** a candidate. Only deterministic corpus-bound exact byte verification, coverage checks and human selection (when required) can authorize a verified result. No candidate or model uncertainty may be labeled PROVEN_EMPTY.

## 6. CUDA/VRAM path (HYPOTHESIS, AFTER CPU GATES)

GPU-FM-index research exists. Sequential FM backward search retains a dependency between steps for a single pattern:
\`[l',r') = [C[c]+rank(c,l), C[c]+rank(c,r))\`, where the next character uses the previous \`l,r\`.
**Block decompression or ACEAPEX CUDA does NOT remove this dependency.**

GPU opportunities: independent queries and blocks in parallel; batched rank requests; carefully designed k-step tables, coalesced compressed-block decode, bulk candidate filtering, bitmap intersections and possibly validated materialization. A single small query may be slower on GPU once launch, transfer, synchronization and verifier costs are counted.

**VRAM is a cache/execution tier, not automatically the durable canonical store.** A PB-scale corpus cannot be entirely resident in finite VRAM; candidate roots and identities stay anchored to durable immutable storage. GPU crashes, stale copies, ECC events and synchronization failures must fail closed. Decoder output must be re-checked against committed hashes by an independently trusted verifier (CPU or separately validated device path). The model has no verification/selection authority even when colocated with GLYPH.

**Persistent kernels are NOT a prerequisite.** First compare: scalar/AVX CPU, ordinary CUDA launches, CUDA Graph replay (reduces repeated launch overhead), and only then a bounded cooperative persistent-kernel queue if proven valuable. Persistent kernels can monopolize resources/SMs and complicate coexistence with inference, isolation, cancellation and fairness. Use measured p50/p95/p99, QPS, bytes transferred, VRAM footprint, Joules/query and end-to-end accepted proof rate rather than kernel-only timing.

References:
- https://pubmed.ncbi.nlm.nih.gov/26451818/ (GPU FM index and k-step mitigation already explored)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html

ACEAPEX CUDA compatibility is a separate unverified integration claim. The historical CPU ACEAPEX block/region-read behavior does not by itself certify any CUDA implementation or a 16 KiB block proof boundary.

## 7. Statistical inconsistency to repair BEFORE using unique-prefix probes

Provided August 2026 Pizza & Chili English 50MB / 100,000-position report:
- \`U(p) = min{L: occ(C[p:p+L]) = 1}\`;
- "Median 13, p90 57, p95 139, p99 230";
- cumulative across all 100,000 positions: <=16 B = 37.739%, <=64 B = 47.106%, <=128 B = 49.112%, <=256 B = 51.994%; 48.006% still non-unique after 256 B.

If percentiles and cumulative figures describe **the same unconditional population**, they are **mathematically inconsistent**: <=16 B cannot be 37.739% if median is 13; <=64 cannot be 47.106% if p90=57; p99=230 cannot coexist with 48.006% censored above 256.

**Likely reconciliation, not yet independently validated:** quantiles may have been computed only for the 51,994 *resolved within 256B*, while cumulative percentages used all 100,000 positions. Conditional resolved fractions:
- <=16: 37,739 / 51,994 = 72.583%;
- <=64: 47,106 / 51,994 = 90.599%;
- <=128: 49,112 / 51,994 = 94.457%;
- <=256: 100%.

These conditional figures are compatible with median=13 / p90=57 / p95=139 / p99=230 in broad terms, but require raw data to verify. Until then label \`mean/quantiles conditional on resolved within 256B\` *if confirmed*, and represent 48.006% as right-censored, not as observed exact \`U(p)\`. Re-run with recorded RNG seed, corpus SHA-256, replacement policy, query semantics and full raw distribution before publishing U-statistics or deriving index bounds.

## 8. Preconditions and staged decision tree

| Gate | Execute after | Pass condition | Kill/suspend condition |
|---|---|---|---|
| T0 Proof baseline | Existing frozen V1, V2, 30-object RLB3X evidence | Strictly reconcile full scope and preserve P/R/O/OVH facts | Claim conflates differing corpora, ignores evidence or calls novel without review |
| T1 Unique-prefix/profiling | Raw sample/corpus recorded | Correct resolved/censored conditional denominators and reference oracle | Distribution fails self-consistency |
| T2 SIMD rank benchmark | Current native/legacy source with unchanged formats | Full count/locate parity and observed speed/storage/RSS win vs scalar on pinned workload | Regression or no practical advantage |
| T3 Composition semantics | Composition 25 EXACT + independent evidence obligations accepted | Strict global document IDs, completeness, root/corpus identity, hostile mutations, no false PROVEN_EMPTY | Any partition-induced semantic difference |
| T4 Field masks | T3, canonical metadata/ACL schemas | Correct filter intersection, cryptographically bound masks, no leakage/stale result | Unbound metadata or false negative |
| T5 GPU batch research | T2/T4 CPU baseline and real GPU with budget | Verified exact parity, full proof cost measured, sustained end-to-end advantage vs CPU/CUDA Graph | Launch/copy dominates, memory exceeded, proof weakens |
| T6 Scale & qualification | T3/T4 and proof-carrying selected restore | 250MB→100GB→1TB→10TB with separate receipts, full bytes, peak RSS, p99 and crash recovery | Extrapolated TB acceptance, unverifiable history, loss of acknowledged bytes |

Research does NOT imply all gates can be pursued in parallel. The weakest missing invariant always gets priority over more hardware features.

## 9. Frozen boundary and immediate decision

DO NOT touch the historical \`rlbwt-binary-safe-v2\` branch, the frozen Laptop/OVH evidence, current \`main\`, or pending PR #15/#17. This document is a research pre-registration, not an E1 ABI freeze, Index Forest runtime implementation, CUDA release or commitment to a proprietary format.

**Recommended next executable action:** semantic proof for filtered multi-block result **before** C++/GPU acceleration: a tiny independent fixture of 3 immutable whole-document blocks with duplicate/empty docs and a field bitmap, plus two different partitions of the same ordered corpus. Verify identical count, canonical locate, filtered counts, externally anchored corpus identity, and fail-closed on omitted/stale mask/block. Test a real match, absent pattern, and a blocked/authorized document. If false-empty or coordinate drift occurs, stop and revise composition/filter semantics, not the proof obligations.

Only after that should we benchmark SIMD; CUDA should be much later and must win end-to-end under the same proof contract.
