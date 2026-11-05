# CORE-L2 — Real, Pinned enwik8 Searchable-Index Comparison

Date: 2026-11-05
Status: **EXECUTED EXPERIMENTAL PASS** on two independently fetched copies of canonical enwik8, with **different trust envelopes**. No production release or universal advantage claim.

## Exact executed sources

GLYPH:
- Experimental HEAD \`b5a90b084b027834a2f78311ca006acd1d8f4e5f\`.
- Frozen source checkout \`rlbwt-binary-safe-v2\` at \`c33cb493bc75f13ee4a904fabb190a29a196c94b\` (not modified).
- [GitHub Actions run 42094185156 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42094185156).
- Two-report CI artifact ID \`13446083265\`.

External:
- Experimental HEAD \`cc3374ef59e52d43c23ac2711c9df7eb3eec26c9\`.
- SDSL-lite source \`c32874cb2d8524119f25f3b501526fe692df29f4\`, pinned libdivsufsort submodule \`0f24acd8de208464769c782119dacf158647f7ed\`.
- [GitHub Actions run 42095220770 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42095220770).
- Two-report CI artifact ID \`13448140369\`.

Independent public corpus download: https://mattmahoney.net/dc/enwik8.zip (exact source length 100,000,000 B; SHA-256 \`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8\`). Each run verifies source bytes and hash. Enwik8 has no embedded NUL: the SDSL classic 8-bit sentinel does not generalize this baseline to arbitrary binary files.

## GLYPH V2 original LOC1 vs experimental LOC2

Canonical enwik8 SA, BWT, RLB2, RLR2 and LOC1 were all **regenerated and matched their historically frozen file-size/SHA256 identifiers** before LOC2 was evaluated. Same RLB2/RLR2/FM query implementation and exact source for both locate cores.

| Component | Historical LOC1 | Experimental LOC2 |
|---|---:|---:|
| RLB2 bytes | 73,365,105 | 73,365,105 |
| RLR2 bytes | 12,844,028 | 12,844,028 |
| locate file bytes | 12,500,040 | **3,125,036** |
| manifest bytes | 828 canonical V2 | 679 explicit research binding |
| **accounted index+metadata bytes** | **98,710,001** | **89,334,848** |
| ratio vs 100,000,000 B | 0.98710001× | **0.89334848×** |

Physical *locate* saving: **9,375,004 bytes**. Net retained-state saving for these separately versioned manifest profiles: **9,375,153 bytes**, because the research manifest is 149 bytes smaller than the frozen canonical V2 manifest. This is NOT proof that the new LOC2 bundle has all original canonical/portable proof contracts.

Query execution:
- 9 exact byte patterns × 4 caps (\`-1,0,1,2\`) × 5 repetitions = **180/180 parity checks**, including source-byte oracle, exact complete locate, valid FM-row-bounded subsets, identical FM intervals and identical LF paths.
- 45 warm complete-query timing samples per candidate: LOC1 p50 **598.501927 ms**, p95 **1,916.792001 ms**; LOC2 p50 **597.825735 ms**, p95 **1,915.912447 ms**.
- Same-process runtime open: LOC1 **0.059533501 s**, LOC2 **0.058824578 s**.
- Maximum *observed* LF path in this workload: **200**. **This is not a global LF bound**.
- Small same-source mutation check: changing a LOC2 byte violates its separately recorded source hash. It was then restored; this test does NOT establish hostile-parsing resistance for a new canonical LOC2 format.
- LOC2 full sample verifier checks canonical SA at build qualification, but an independent verifier should not need to retain full 400 MB construction SA just to validate runtime query claims.

**Critical limit:** LOC2's current direct Python wrapper uses explicit paths, unlike the formally versioned canonical V2 LOC1 manifest binding. The new 679-byte manifest is **research-only** and cannot automatically inherit P/R/O/I0, V1 portable proof or hostile gate acceptance. The measured experimental ratio is not a production GLYPH proof-carrying bundle ratio.

## Pinned independent SDSL baseline

The stock \`sdsl::csa_wt<>\` implementation (not a tuned search of all configurations), compiled from pinned source, builds a serialized exact-search index over the *same canonical enwik8*. A standalone external adapter checks 9 identical patterns against an **independent raw byte-window oracle** and reconstructs all 100,000,000 source bytes from the SDSL index after its original source pathname is inaccessible; final SHA256 equals the canonical source digest.

Measured SDSL serialized **index-only**: **110,670,771 bytes**, ratio **1.10670771×**. A separate research identity manifest was created and measured, but no GLYPH-equivalent cryptographic proof bundle was attached.

Comparison on a SINGLE corpus, with explicit feature differences:

| Representation | Serialized stored bytes | Exact count/locate | Source byte extract | Independent root/evidence proof |
|---|---:|---|---|---|
| GLYPH V2 canonical LOC1 | 98,710,001 | historical V2 tests + new 180 parity cases | not a standalone new source-restore qualification in this test | manifest SHA binding; not full V1 trust |
| GLYPH LOC2 research | 89,334,848 | 180 parity cases | not new extract proof | experimental SHA/research manifest only |
| SDSL-lite default \`csa_wt<>\` | 110,670,771 index-only | 9/9 external oracle | 100MB byte-restore SHA PASS | no GLYPH proof/evidence layer |

Differences versus this **one SDSL configuration**: historical GLYPH V2 index is **11,960,770 B smaller**; research LOC2 index+manifest is **21,335,923 B smaller**. These are useful facts, not evidence that GLYPH is smaller/faster than every SDSL or r-index configuration. SDSL has more advanced CSA variants/sampling choices, and the benchmark runs use separate GitHub-hosted runners. No statistically valid speed comparison between cross-run systems is claimed.

## Independent second enwik8 SA-only qualification

To avoid relying solely on the long combined Python query run, a **separate clean GitHub Actions job** rebuilt the 100MB enwik8 SA from the pinned historical compiled builder, matched the historical SA digest, regenerated canonical LOC1 and all **781,251** LOC2 sampled offsets against the full SA. This used no RLB2/RLR2 combined runtime and makes no query-accuracy claim by itself; it is independent corroboration of physical LOC2 geometry.

- [Run 42095705324 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42095705324).
- LOC2 exact SHA256: `4f90560aeccf86e077b888a129f06f14fa655ea1eab1e445c6cb11f26eb6011c`.
- LOC2 bytes: **3,125,036**, checked **781,251** samples; canonical LOC1 SHA `1ba04acce3b2ebea10ae3ab703138c1a33e7c9caa27fe7a1056d4ed8c44f7f6f` and SA SHA `a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727`.
- LOC2 alone saves **9,375,004 B**; source/index SHA, geometry and duplicate artifact generation are now independently reproduced under a separate runner.
- The reference runtime still lacks a canonical signed/externally anchored LOC2 production manifest or full portable proof format; construction-time SA is not retained runtime state.

## Implications and next falsifiers

1. **LOC2 has now been physically measured on canonical enwik8**, not merely predicted by arithmetic. This is a size/geometry and exact count/locate preservation win, NOT a native speed win.
2. Move LOC2 into an **additive, versioned canonical manifest contract**, without modifying frozen V2 LOC1 or P/R/O/I0. Include file hash, corpus/SA sampling geometry, canonical decoder identity and fail-closed hostile parse gates; then independent source-bound query replay from published bytes with no construction SA.
3. Independently investigate **bounded LF steps**, as the current periodic *BWT row* sampling does not prove \`<=sample_step-1\` LF iterations.
4. Compare additional SDSL \`csa_wt\` sampling configurations or pinned r-index using exactly the same source, query list, correctness rules, memory and complete size accounting. SDSL's own capacity to reconstruct exact source does not imply any GLYPH-level authenticated currentness or owner signature.
5. Keep compression/proof-size claims separate: \`RLB2\` alone is 73,365,105 B on enwik8, so a full **<=0.70×** ratio is impossible under unchanged RLB2 payload for that corpus. Further gain requires a different compressed representation, not merely shrinking LOC2.

The accepted research results are additive evidence, **not permission for PR #18 merge**, Index Forest runtime, E1 ABI freeze, replacement of Elasticsearch/Snowflake, 1 TB acceptance or complete cryptographic nonmembership.
