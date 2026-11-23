# CORE-L10 Gate 0 — compatibility audit BEFORE adapter implementation

Observed PR #27 HEAD: `e7f0e3b16381519c8b0269f6fea0ff11feb0cc80`.
Observed PR #25 HEAD: `7a48c499a0fc601cc6289e5644e3b2db986e7aba`.
Both were Draft/open/unmerged. Binary-safe frozen reference:
`c33cb493bc75f13ee4a904fabb190a29a196c94b`. Historical Composition operator
generation: `81628f843dc8aa633c65594a66d2386bfc3cb5c2`. Distinct generations,
not interchangeable immutable checkpoints.

No algorithm was modified before this audit and the complete original
Composition checker/replay executed successfully in a separate disposable
historical checkout. This document authorizes no normative contract/ABI change.
Only the non-normative `GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0` adapter is
proposed here; executable success still requires the later gates.

| Property | Historical actual API/contract | L9 actual API/contract | Binding / verdict |
|---|---|---|---|
| Block physical runtime | `glyph_operator_index_v1.build_runtime_index`, `verify_runtime_index`: `GLYPH_BINARY_RUNTIME_V1`, `one_independent_index_per_document`, retained SA/BWT/FM per document, source snapshots, runtime/source manifests and complete markers | `query_ef_research.make_runtime`: frozen RankV2 + RLB2/RLR2 + experimental physical EF; no operator runtime manifest | **CONTRACT_MISMATCH**. An RLB2 leaf is not an existing operator unit. L10 block is an explicitly named research grouping of verified leaf bindings, not an old runtime-index manifest. |
| Runtime identity | `GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1`: dense global doc ID, length, raw SHA; document count | L9 source identity is only one corpus length/SHA, no doc ID | Reuse the **exact old global preimage**, from an independently checked ordered document manifest. A single-corpus source SHA is never aliased to corpus ID. |
| Source manifest identity | `GLYPH_OPERATOR_CORPUS_MANIFEST_V1`: count + ordered doc ID, relative path length/bytes, source length/SHA | L8/L9 root has no source-path binding | Reuse exact identity preimage for the valid path subset. Identity is not the SHA of old serialized source-manifest JSON; physical manifest formats remain different. |
| Physical root | `composition_root_id`: `GLYPH_COMPOSITION_ROOT_V1\0`, global IDs, counts, ordered operator runtime IDs/manifest hashes | L9 trusted root is SHA of canonical L8 JSON binding one source + RLB2/RLR2/EF + decoder | **CONTRACT_MISMATCH**. New `research_composition_root_id`, with distinct domain and explicitly versioned body, binds grouping, ordered leaf commitments, code/decoder and parent. Never emit it as old `composition_root_id` or `runtime_index_id`. |
| Coordinates | `(doc_id,doc_offset)`, prefix-summed contiguous block ranges; block identity is provenance | Single source `corpus_byte_offset` only | Exactly one document per nonempty leaf. Leaf's offset maps to its committed global document position; no byte-offset-to-doc-ID guess. Block ranges are recomputed and checked. |
| Boundaries | `DOCUMENT_LOCAL_MATCHES_ONLY_V1`, no physical byte concatenation | One independent FM search domain per leaf | Query every document leaf independently. Grouping never concatenates source bytes; both doc/block-only straddles stay absent. |
| Empty document | Empty document preserved in global identities and later coordinate bases; original operator creates a real per-document SA/BWT/FM even for empty payload | L8/L9 positive source-length contract rejects zero | **CONTRACT_MISMATCH (physical)**. Explicit empty binding, zero length, SHA256(empty), no claimed index/leaf receipt. Independent verifier requires the empty binding and verifies an empty oracle object/hash. Preserves old nonempty-query empty-set semantics, not old physical format. |
| Duplicate contents | Distinct manifest positions/paths remain distinct doc IDs; no document deduplication | Equal source bytes can yield equal L8 root/component hashes | Separate doc-slot binding manifests commit doc ID/path. Two physical leaf directories are retained even when component bytes match. Missing/merged slots fail coverage. |
| Full coverage | `expected == verified == queried` for every root block, all mandatory artifacts, no partial success | All source/rank/locate/root mandatory; count=0 does not establish coverage | Root commits physical block manifests and exact leaf slots. Verify/query all before any trusted global result, including zero. Empty slot is explicit, never inferred from missing nonempty data. |
| Count / caps | Count sums over all blocks; one global canonical prefix, max_offsets 0 valid | L9-A/B full leaf receipts <=256 hits; limited is UNTRUSTED; schema V0 | Keep old leaf receipts unchanged. Initial L10 doc cap 256B ensures every nonempty exact leaf count <=256. Sum all leaf counts, then apply **one global** cap 0..256. Full coverage and complete offset enumeration are separate flags; incomplete enumeration remains UNTRUSTED. |
| Independent replay | Original `replay_composition_reference_v1.py` imports operator verifier/query, checks physical blocks, independently scans source snapshots, reruns actual queries | Small L9 verifier is a standalone stdlib file, derives bounded SA/BWT oracle after source SHA; <=4096B. L9-S verifier streams two source passes, <=100MB canonical enwik8, structural-only index checks | Initial L10 independent verifier has no composition-producer import; independently rebuilds identities/count/coordinates from hash-checked per-doc oracles and separately invokes unchanged small L9 verifier on each actual nonempty leaf receipt. Native L8 produces ephemeral oracles outside query state. No claim of 100MB multi-doc replay. |
| Append | Historical reference defines immutable roots/repartition; does not implement append history protocol | L9 roots bind a single immutable source | New non-normative generation 0/1 roots: 9 ->12 documents; old slot/binding/file bytes identical, IDs stable. Verify exact old/new ordered manifests and leaf-binding prefix using a separately pinned expected old root. Old-root link alone is not a consistency proof. |
| Authority | Digest identities are not author/currentness signatures | Explicit unsigned external-root assumption | Remains unsigned, no rollback/fork/freshness witness. V0 receipt is never interpreted as V1. |

## Exact reused preimages (not new aliases)

Runtime corpus ID:

```
SHA256("GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1" || 00 || U64_BE(N)
       || for i in order: U64_BE(i) || U64_BE(length_i) || raw_sha256_i)
```

Source manifest ID:

```
SHA256("GLYPH_OPERATOR_CORPUS_MANIFEST_V1" || 00 || U64_BE(N)
       || for i in order: U64_BE(i) || U64_BE(path_bytes_i.length) || path_bytes_i
                         || U64_BE(length_i) || raw_sha256_i)
```

The adapter initially accepts a bounded safe-ASCII canonical relative-path
subset, unique and sorted by bytes, satisfying the original path rules. It
does not broaden the original file domain or claim complete operator-manifest
serialization compatibility. P4 name-bearing proof `corpus_id` is a different
profile and is not replaced.

The first nine documents will be **exactly the historical checker fixture**,
so genuine global identity agreement can be compared directly with independently
executed old code, not merely between two copies of new hash functions.

## Executed historical evidence (local Gate 0)

Commands, cwd at immutable Composition checkout (builds/temp only):

```
cmake -S . -B build
cmake --build build --target build_sa_binary_v1 build_bwt_binary_v1 \
  build_fm_binary_v1 query_fm_binary_v1 query_fm_locate_binary_v1 -j2
python3 -I tools/check_composition_reference_v1.py
```

Exit 0. Final `GLYPH COMPOSITION REFERENCE OK`; **25 EXACT / 0 OPEN**, 34
implemented mutations. Original checker itself spawned the original independent
replay from two different working directories, required deterministic output
and marker `GLYPH COMPOSITION INDEPENDENT REPLAY OK`, then rejected three
independent replay negatives. No checker/replay edits or mocks.

Observed global runtime corpus ID:
`4765ce3496058719db1436c56a33b325b16dc1c4cdfecd86664d0bdfb94dc950`.
Observed source manifest ID:
`a21e00fea40c4f7fb31b22c86400436acd90dba6e6efba855b4395ce7d162e09`.
Local old physical roots: A
`0df315ca047a7b6700b1aac338553223370b8b4a9b177d3b69fa2aafc6dda7d7`,
B `7b79882d4a853a9dee4166d43f9547221cd78637cac999c30a6d370bfb168c04`.
These physical roots include actual compiled binary commitments and are not
hardcoded cross-toolchain expectations. The two global document identities are.
Original checker stdout SHA256:
`fbcf2d4360c63df38a68f7041a2f713125d9f186129504638371facf3cc1a8c3`.
Historical tracked tree remains clean.

## Audited source identities

All values are SHA256 of actual files, before adapter coding.

| Checkpoint / file | SHA256 |
|---|---|
| Composition spec at 81628f8 | `9534a79eeb5608eefdb9fddc682ebd65904bf70c28c42ffac1af2a47f5016f59` |
| `tools/check_composition_reference_v1.py` | `3ce11f0694be3e832bbbce540bb5d51cda90449010c24506422f0a8044fc887f` |
| `tools/replay_composition_reference_v1.py` | `62512bee60a2f03d7d2c2680a0d84baa1a66240b9dd3dafa5051719dcb5b66aa` |
| `tools/glyph_operator_manifest_v1.py` | `55dc0599893b5e0b8f133fb98bcce9cb65e17fea6439c5cb131c0aea526a4012` |
| `tools/glyph_operator_index_v1.py` | `03cb5526afa4ea2dbf8d242a0fdfd72bf0489ba4ef349273b20110fc8f31d7e9` |
| `tools/glyph_operator_query_v1.py` | `47826508ed6e05a65aa3309fcd1a303ac3b97a10711f98ddb7b375ad87a46c39` |
| L9-A/B `producer.py` at e7f0e3b | `56bbf9785db87dc529699b8db0158de444e373f3b5f27a82aadfc3b5d87ebc49` |
| L9-A/B `verifier.py` | `5ec4df94e7d5f0246f494aa233bbb653487db7445f84743b6c0b2824169a5333` |
| L9-S `producer.py` | `71b2ed9eb151d9c9a1b800696746a1154ccfe5d0489ef085094f56c6bae0ed29` |
| L9-S `verifier.py` | `0f6a05f80398107f8e61b936e98a0783af7f8512ff428a73325c977922805850` |

## Prior art and deliberate nonclaims

[Papadopoulos et al., PVLDB 2015](https://www.vldb.org/pvldb/vol8/p750-papadopoulos.pdf)
studies authenticated pattern matching using a different cryptographic
construction. This adapter makes no novelty or optimal/succinct-proof claim.
[RFC 9162 §2.1.4](https://datatracker.ietf.org/doc/html/rfc9162#section-2.1.4)
defines Merkle consistency verification between tree heads. The proposed full
ordered-prefix replay is **not** that Merkle protocol, uses no CT tree-head
serialization/signature and is not called a cryptographic consistency proof.
The DOI/RFC-editor endpoints failed during reading; the primary publisher PDF
and IETF datatracker text were consulted instead. No code is copied from them.

Gate 0 conclusion: truthful identity/coordinate binding exists for the bounded
fixture. Physical **CONTRACT_MISMATCH** is explicit and remains non-normative;
old operator-root compatibility is neither implemented nor claimed. Proceed
only with the additive adapter and its own tests/workflow. No Index Forest,
normative ABI, historical format mutation, merge or production operation.
