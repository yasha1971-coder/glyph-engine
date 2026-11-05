# CORE-G1D — Real Operator Composition vs Independent Byte and CRoaring Oracles

Recorded: 2026-11-05
Status: **SCOPED REAL-OPERATOR SEMANTIC COMPARISON — EXECUTED PASS**.
Verified CI: https://github.com/yasha1971-coder/glyph-engine/actions/runs/42091174565
Exact executed experimental HEAD: \`8f727bdee761669567a5589b2f98c60d4a897a95\`
Pinned unmodified GLYPH Composition Reference checkout: \`81628f843dc8aa633c65594a66d2386bfc3cb5c2\`.
External candidate: CRoaring via pinned \`pyroaring==1.2.0\`, installed in isolated GitHub Actions runner.
Producers: \`experiments/core_convergence_v1/real_operator_comparable_gate.py\` and \`.github/workflows/core-real-composition-external-oracle.yml\`.

## Exact implemented experiment

On fresh Ubuntu CI, compile **existing C++** \`build_sa_binary_v1\`, \`build_bwt_binary_v1\`, \`build_fm_binary_v1\`, \`query_fm_locate_binary_v1\` from the pinned historical Composition checkout.

Build actual \`glyph_operator_index_v1\` runtime units for the *existing* nine-document Composition reference fixture with immutable binary, duplicate and empty documents, in two alternative 3-block layouts:
- A = [0,3) [3,6) [6,9).
- B = [0,2) [2,6) [6,9).

Each has physical manifests and runtime artifacts generated through original \`build_root\`/\`build_snapshot\`/\`build_runtime_index\`; no mock block identities were used. Commit exact root manifests for this disposable runtime and query the full block set through the original operator implementation. Validate with the original Composition oracle at build time. Independently in fresh \`python -I\` processes, run **the original** \`replay_composition_reference_v1.verify_root\` and \`verify_result\`, reexecute compiled query and its raw-byte oracle, then verify an additive versioned field-mask sidecar against the *verified document bytes and source manifest identity*. CRoaring performs an independent document-ID intersection and output is checked again against the raw document-local oracle.

Eight exact patterns per partition, including \`shared\`, \`dup\`, binary \`00ff\`, absent and \`00ff736861726564\` cross-document trap; 16 full query replays. Additional 12 filtered results per layout: 24/24 filter claims checked, including bounded offset lists (0 and 2). No cross-document matching is allowed. One deterministic mask is bound to the same logical corpus and source identity in both partitions, but its two externally supplied trust-pin hashes differ because physical roots differ.

## Result IDs

- Global runtime corpus ID (A and B):
  \`4765ce3496058719db1436c56a33b325b16dc1c4cdfecd86664d0bdfb94dc950\`
- Physical composition root A:
  \`d7e755d8d50dedac265be0e0091269c044b84876ab6aa00a914a8191e50c5984\`
- Physical composition root B:
  \`a02ca48bada47c30feca842ae8b60979195983ba0d9b8311697b00b73d3a9014\`
- Logical field-mask ID (same A and B):
  \`f99597a6f17d2ec10e3b9c5188ef823ae1f64542c7220e10246e98bb999e7fb5\`
- Independent semantic filtered result set SHA-256 (same A and B):
  \`fb91472f3611962d1ea3589c8cb95da1ecba379df6b140600636f92d64a699e2\`

The five mutation runs were each rejected in an independent verifier process:
1. One required physical runtime block unavailable.
2. An out-of-date/mismatching out-of-band pin.
3. A stale/replaced field mask generation.
4. Mutated root manifest bytes.
5. Claimed \`PROVEN_EMPTY\` with all coordinates deleted, match count set to zero, and the receipt's own hash **recomputed**.

Result marker:
\`SCOPED_REAL_OPERATOR_COMPARABLE_PASS_NOT_PRODUCTION_FOREST\`.

GitHub Actions job \`comparable-real-operator\` SUCCESS (both pinned checkouts and actual C++ build succeeded). The job reports its bounded \`result.json\` and uploads only this file as its Actions artifact, not temporary corpora.

## What this means

The exact operator/Composition **proof-carrying semantics** can be extended in a separate research adapter to a field mask without allowing a false empty result, in the scope of this reference fixture. The experiment tests the **actual historical compiled runtime artifacts**, external CRoaring mask semantics and an independent byte oracle rather than a newly invented FM index.

**This is NOT a delivered Index Forest runtime**: the mask sidecar/filtered receipt are new experimental forms; they are NOT added to versioned P/R/O/I0 or the existing O1–O6 portable bundle, not a signed remote field-authority witness, not authenticated general cryptographic non-membership, not tested under adversarial concurrent writers or policy/ACL isolation. The CI trust pin is passed separately by the same test driver; it is not protection against a hostile authority rewriting that pin. The independent verifier is a separate process, but uses source code in the pinned Composition checkout: no novel fully standalone cross-host field-proof bundle claim.

The corpus is tiny and fixed: no size, latency, RSS, SIMD, GPU, 100GB/1TB, or comparative r-index/SDSL performance is established. Correctness of unfiltered exact substring count/locate is measured on eight patterns per partition, not exhaustively quantified for all patterns and documents.

## External comparison candidates and next gate

The accompanying [EXTERNAL_BASELINES_AND_ORACLE_POLICY_V1.md](EXTERNAL_BASELINES_AND_ORACLE_POLICY_V1.md) identifies **r-index (Nicola Prezza)**, **SDSL-lite \`csa_wt\`**, **CRoaring** and **Pizza & Chili repetitive corpus** as grounded, relevant external baselines; only the CRoaring field-semantic comparison was **executed** in this gate. The r-index CLI discards resulting offsets, requiring an output-capable external adapter or source instrument before it can be an independent correctness baseline. Never claim it has been run.

**Recommended next strict gate**: real, immutable, portable Composition field-mask proof replay under independently pinned currentness, with external verifier and full proof storage accounting. In parallel only as an isolated performance experiment, benchmark LOC1 vs pre-existing LOC2 on the exact enwik8 bytes and compare to a pinned SDSL implementation with equivalent count/locate semantics; reject incompatible sentinel or document-boundary handling.

Keep PR #6 composition semantics and P/R/O/I0/frozen V1 untouched. This result is additive to draft PR #18 and is not permission to merge or implement Index Forest runtime.
