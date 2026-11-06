# CORE-LF / LOC-TEXT-EF — Executed Research Evidence, 2026-11-05

**Status:** Three CI gates EXECUTED PASS. Mathlib typecheck is a separate gate with separate status. **Not a frozen runtime, release, or scientific novelty claim.** Frozen historical source: \`rlbwt-binary-safe-v2\` \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`.

## GATE A — exhaustive small binary / full LF oracle: PASS

[CI run 42111369116](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42111369116). \`exhaustive_binary_lf_sampling.py\` explores all **9,841** raw binary corpora of lengths 0 through 8 over \`0x00,0x01,0xff\` and **77,092** corpus × step configurations. The suffix array uses the original \`byte+1\`, sentinel \`0\` ordering. For every SA row it checks the original offset after physical text sample lookup and LF walk, while LF is independently reconstructed via BWT's C+rank as well as inverse SA.

- Text-sampled candidate \`SA[row] % s == 0\`: **0 LF-bound and 0 offset failures**; the mathematical \`s-1\` LF limit holds for all tested rows.
- Historical row-periodic sampling: **34,314** small corpus × step configurations exceed that limit. Example \`raw_hex=000100\`, \`s=2\`, old max LF=2, new max LF=1. These are valid counterexamples to the claim of a 1-step limit for row-periodic SA, NOT evidence of wrong historical returned offsets.
- Stronger theoretical proof obligation is simple arithmetic plus an accurate rank-based LF permutation. Mathlib typechecking is tracked separately.

## GATE B — actual enwik8 file, fully counted physical bytes: PASS

[CI run 42111671020](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42111671020). Independent GitHub Actions runner downloaded canonical enwik8, SHA-256 \`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8\`, compiled and ran *unchanged* original C++ SA builder at the pinned commit. Canonical 400,000,064-byte SA SHA-256 was exactly reproduced: \`a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727\`.

Original SA row count: 100,000,001; sampling step 128; **781,251** samples. Real \`GL3EF001\` research file uses strict Elias–Fano monotone SA-row set, bitpacked \`SA/s\` quotients, checkpoints every 256 sampled rows, and versioned geometry header. It was independently read and **every one of 781,251 sampled SA values** matched the untouched original SA.

| Accounted physical component | Actual bytes |
|---|---:|
| EF low values | 585,939 |
| EF high bitvector | 292,969 |
| 20-bit \`SA/128\` quotients | 1,953,128 |
| 32-bit select checkpoints (every 256 samples) | 12,208 |
| Research header | 80 |
| **Total new research locate file** | **2,844,324** |
| Historical experimental LOC2 file | 3,125,036 |
| **Difference** | **280,712 bytes less** |

New file SHA-256 \`c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7\`. The preregistered physical size prediction was **2,844,324** and the measured value exactly agrees; this validates the geometry/packing model on this corpus. Two invalid-header/geometry cases were also checked. *This alone does not validate an independent source-restore or trust bundle.*

## GATE C — actual frozen RLB2/RLR2 RankV2 with physical EF reader: PASS

[CI run 42112293509](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42112293509). Original \`check_rlbwt_query_v2.build_fixture\` generated actual RLB2/RLR2 bytes and historical LOC1/LOC2 samples from a **24-byte binary** corpus containing embedded NUL, 0xFF and repeated substrings. The candidate read **its physically packed EF bitstreams and physically stored select checkpoints**. This was NOT a Python-only fabricated mapping table.

- **25/25** SA rows resolved to the same exact byte offsets as canonical SA and original LOC1.
- For \`s=2\`, maximum candidate LF steps **1**; old row-periodic LOC1 maximum **3**.
- **12 patterns × 4 max-offset caps × 5 repetitions = 240/240** count/FM interval/offset cases PASS against independent source-byte oracle. Full locate list exactly equals source-order oracle; bounded cases are exact valid FM-row subsets, and O3 source-order prefix semantics remain a separate unfinished API issue.
- **2 hostile scope checks**: externally pinned SHA mismatch after payload mutation, malformed header rejected by actual reader.
- **Critical limit:** an external digest must be supplied by a trusted authority; the current research reader does NOT authenticate itself and does not prevent an attacker from rewriting both index and external pin.

## Formal math / worldwide prior art

Sampling *by text position* to bound LF steps is classical compressed suffix array/FM-index engineering. The scientific comparator must include **Fast and Small Subsampled R-indexes** (ACM Transactions on Algorithms, 2025), **Optimal-Time Text Indexing in BWT-runs Bounded Space** (Gagie, Navarro, Prezza), and contemporary compressed SA/ISA literature. https://doi.org/10.1145/3750729 ; https://arxiv.org/abs/1705.10382 ; https://arxiv.org/abs/2607.17287 . This candidate is a specific executable composition of classical structures with an exact, counted file and an unusual *quotient-coding space tradeoff*; **not a claim that Elias–Fano sampling was invented here**.

\`BoundedLFSampling.lean\` formalizes the modular arithmetic contract under pinned mathlib4 \`v4.25.0\` \`1ccd71f89cbbd82ae7d097723ce1722ca7b01c33\` and Lean \`v4.25.0\`. The Lean CI verdict must be obtained independently; a checked arithmetic lemma does not formally verify the RankV2 or C++ byte parser. \`openai/math\` is public experimental mathematics with not all claims formalized, not a proof oracle.

## Proof-accounting boundary / structural impossibility

Canonical enwik8 original RLB2 alone occupies **73,365,105 bytes** (0.73365105×). Therefore, a **fully retained authenticated <=0.70× state is impossible on enwik8 if RLB2 is unchanged**, regardless of how aggressively LOC2/LOC-TEXT-EF is compressed. Saving 280,712 locate bytes improves a narrow measured Pareto component, not GLYPH's main compression gap. Further savings below 0.70 require a new entropy representation and/or changing rank/locate representation, with full proof/byte restore and decode costs counted.

Candidate budget before any **new** manifest/proof/verifier code/payload:
\`73,365,105 (RLB2) + 12,844,028 (RLR2) + 2,844,324 (LOC-TEXT-EF) = 89,053,457 bytes\`, ratio 0.89053457×. **This is an arithmetic budget, not a certified self-contained new release.** User-facing GLYPH must retain exact query, proof evidence, canonical document coordinates, actual source extraction, atomic generations and fail-closed corrupted-data behavior.

## NEXT STRICT GATES

- D: actual executable Lean/Mathlib typecheck, no \`sorry\`, no unreviewed runtime axiom; status separate.
- E: new file's physical \`select\` and rank/locate native performance at full enwik8, worst-case LF steps and p50/p95/p99, RSS and decoded bytes; compare LOC1, LOC2 and this candidate on same machine with same prepared query list.
- F: reconstruct all original source bytes from the *same complete retained search state*, hash-verify, account any extra source/payload once. A reversible BWT is theoretically self-indexing but the required full inverse path and efficient bounded extraction must be implemented and measured; do not treat the 24-byte example as 100MB proof.
- G: independent canonical manifest, external committed root, corruption/rollback/false-empty hostile tests, P/R/O portable receipt and verified byte-range evidence. No source-only build data in the package; enforce lifecycle/canonical document ordering and human-gated materialization for Personal Vault where relevant.
- H: use ACEAPEX only as separately measured optional extractor until the *entire* retained dual representation and trust model are measured. No free double-counting or claims that a separate archive creates a single physical compressed index. No GPU work before CPU and proof equality.

**Owner gate:** No merger to historical \`rlbwt-binary-safe-v2\` or \`main\`, no ABI freeze, no O3/O4 schema rewrite and no deploy without explicit strategic approval. All research is additive and reversible.
