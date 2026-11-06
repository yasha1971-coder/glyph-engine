# CORE-L10 — immutable append research (RLB2/RLR2/EF, not Composition V1)

**Gate status: EXECUTED RESEARCH PASS (tiny deterministic corpus), not frozen GLYPH V1 compatibility.** Source commit: `f77a6d05f54f4349a42361696d54f759e42320fe`; CI [42198842880](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42198842880) `SUCCESS`.

Reference sources:
- Composition Reference V1 `81628f843dc8aa633c65594a66d2386bfc3cb5c2`.
- Frozen binary-safe encoder `c33cb493bc75f13ee4a904fabb190a29a196c94b`.
- L9-S predecessor `e7f0e3b16381519c8b0269f6fea0ff11feb0cc80`.
- [Existing task Issue #28](https://github.com/yasha1971-coder/glyph-engine/issues/28). No further GitHub Issue needed.

## Results independently executed

The GitHub Actions workflow first checked out the exact historical Composition commit and ran its untouched `tools/check_composition_reference_v1.py`, which itself uses the independent Composition replay, returning `normative_exact_count=25`, `normative_open_count=0`, `normative_closure_complete=true`, independent replay PASS.

Next, the new *different, nonnormative* research adapter built actual frozen **RLB2 and RLR2** with the original pinned encoder and rank builder, and constructed/independently compared **physical GL3EF001** text-position samples to the fixture suffix array. One physical RLB2 index corresponds to exactly one nonempty document. The one empty document is explicitly hashed and represented as a manifest-only document, consuming one stable `doc_id`. The fixture source is defined in the test code and is **not retained as a second compressed archive**.

**V0:** 9 ordered logical documents, 8 indexed nonempty leaves and 1 explicitly hashed empty document. Two reference-style grouping arrangements 3+3+3 and 2+4+3 use the same 9 leaf indices. Both layouts have EXACTLY the same normative `runtime_corpus_id` and `source_manifest_id` (original V1 preimage rules), and identical sorted canonical `(doc_id,doc_offset)` responses. Their distinct physical groupings have **different `research_layout_id`**.

**V1:** Three new documents were appended physically, making 12 logical documents and 11 nonempty indexed leaves, with new canonical `doc_id=9,10,11`. No old leaf was encoded again during the append step. **24/24 old indexed component files** retained exact original SHA-256 and byte sizes before/after append. Every former source/doc identity and coordinate was retained. V0 root manifests stay pinned, independently replayable with their original known SHA. V1 manifests explicitly pin the preceding respective V0 root and have distinct new runtime corpus/source manifest IDs. The literal `V1_NEEDLE` query is `PROVEN_EMPTY` for complete verified V0 and `FOUND` only in V1 at `(doc_id=9,doc_offset=0)`.

All roots and results are strictly, canonically serialized JSON under an explicitly new research schema. Source materialization for each nonempty leaf in the independent verifier goes through the original frozen **RLB2→BWT** decoder, followed by **separately written Python LF inverse** (NOT a producer import). For this tiny fixture, the verifier builds an independent source suffix array and checks every EF sample+checkpoint+padding against its own independently reconstructed bytes. The producer instead uses real RankV2 and physical EF locate/extract, authenticating every indexed leaf before any global answer and enumerating complete physical offsets for this bounded test. Missing coverage or corruption returns `UNTRUSTED` with nonzero exit.

**Measured CI result:**
- 36 actual fully and independently replayed multi-document queries across both V0/V1 physical layouts and global limits 0/1/2/full.
- 37/37 hostile cases rejected, including 25 altered result fields/offsets, 10 invalid/manipulated roots, one missing required RLR2, and one **re-pinned EF select checkpoint** zero-hit producer regression.
- Indexed RLB2+RLR2+EF data across the total 11 nonempty leaves: **42,495 bytes** on this specially small synthetic fixture. This **excludes** root files, receipts, code, Python interpreter, metadata, verifier, reference checkout and temporary filesystem snapshots; do NOT promote it as full installed package ratio.
- Entire V0->V1 append and replay CI returned marker `CORE_L10_APPEND_RESEARCH_EXACT_AND_INDEPENDENT_REPLAY_PASS` and exit 0; original Composition executable gate marker `CORE_L10_HISTORICAL_COMPOSITION_REPLAY_PASS` also exit 0.

## Crucial retained incompatibilities

1. **NO Composition V1 operator-runtime compatibility.** The old Composition requires real `GLYPH_OPERATOR_RUNTIME_INDEX_V1` blocks with complete O1/O2 snapshots and manifests; an RLB2/EF leaf is not that block. Root field `research_layout_id` is **NOT** `composition_root_id`. Calling the former the latter would be a contract violation. See [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md).
2. **No arbitrary file-path append with stable IDs.** Old V1 source manifest sorts raw path bytes. The test uses a monotonic synthetic namespace `vault/00000000.bin` onward. To support actual user paths arriving at any time, a separate stable internal document identifier/ordered-ingest manifest policy needs normative owner approval and proof of mapping to actual source path metadata.
3. **No signed append history or trusted latest root.** V1's predecessor root SHA and byte equality are replayable under externally supplied pins, but do not prevent rollback/forks or claim currentness. No RFC 9162 consistency proof exists in this new profile.
4. **Not independent O(1)/succinct absence proof.** The tiny verifier restores every document and builds an O(n²)-style suffix comparison oracle; no 100MB multi-document run was performed. A missing mandatory leaf rejects output, never claims a globally negative result.
5. **No universal production Index Forest, no C ABI, no 1TB/64-bit, no full P/R/O source+signature evidence.** The research adapter keeps distinct IDs and never edits historical Composition code.

## Engineering priority after this gate

The next blocker is **normative identity and operator compatibility**, not writing yet another FM-index core. To make user-facing daily append real, define a separately versioned immutable stable-document-ID/ordered-ingest namespace, source-path metadata reconciliation, true per-leaf operator evidence contract and independently anchored snapshot lineage. Only then implement multiple persistent segments/index roots with independently verified complete query coverage. Test at real corpus sizes and account *all* retained bytes and temporary disk.

**Owner veto:** no merge, main, frozen, historical composition checkout, public demo, OVH, ABI or release changes. Draft research only.
