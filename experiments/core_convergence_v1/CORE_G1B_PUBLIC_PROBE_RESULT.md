# CORE-G1B — 30-object public V0 proof-carrying reproduction

Recorded: 2026-11-05  
Status: **SCOPED V0 PROOF PASS; UNIFIED P/R/O/V1 PORTABLE PROOF INTEGRATION REMAINS OPEN**  
Execution commit: \`8b962abf40dbfa407c1ae99c2888b028562c816c\`  
Successful GitHub Actions: https://github.com/yasha1971-coder/glyph-engine/actions/runs/42085257299  
Two-report artifact: https://github.com/yasha1971-coder/glyph-engine/actions/runs/42085257299/artifacts/13442721359  
Archive ZIP SHA-256 (GitHub Actions): \`ab700c26759592db7267ae7abf6fac3ceddf2df50042ef02d6057d31fa72ae6b\`

## What was executed

The probe ran the repository's EXISTING \`glyph_vault_cli_v0.py\` (init / add / verify), \`proof_carrying_recall_v0.py\`, \`verify_proof_carrying_recall_v0.py\`, RLB3X builder, LOC2 builder, restorer and native SA/BWT builders against the PINNED public source. Original V0 source code and frozen V1 were not modified.

Corpus: zlib-ng/corpora at \`5583ca94d1643b6dcd6b6dd2ad0c5704a4afa094\`. Exactly **30 ordered files**, **7,230,581 logical bytes**, SHA-256 \`4f3fa33cc6276dd8ec1f8099056646b5a63f1e18d2230753618db8813eff6a67\`.

Query: \`Wonderland\`; committed receipt binds \`canterbury/alice29.txt\`; **2/2 object occurrences** were independently compared with original public bytes. The original source directory path became inaccessible during verification by **renaming** it to a held-out location in the disposable runner (not deletion). V0 verifier restored the relevant corpus from compressed RLB3X, recomputed exact object SHA-256, and checked receipt, object offsets and all occurrences. A mutated receipt was rejected with \`VERIFY_FAIL\`.

CI: exit 0, \`SCOPED_V0_PROOF_PASS_FULL_PROOF_INTEGRATION_NOT_CLOSED\`.

## Measured physical bytes (one query receipt)

| Component | Retained bytes |
|---|---:|
| RLB3X compressed self-index (payload+search) | 2,191,468 |
| LOC2 locate | 225,988 |
| objects.json | 4,370 |
| source-state.json | 5,065 |
| segment-manifest.json | 819 |
| committed root manifest | 336 |
| repo.meta | 159 |
| last-verify sidecar (retained diagnostic) | 261 |
| one query's proof-carrying receipt | 1,211 |
| **Accounted V0 Vault + one receipt** | **2,429,677** |

Reported ratio of *this bounded retained-state scope*: **2,429,677 / 7,230,581 = 0.3360279070×**. The previous compact kernel-only result was **2,422,268 bytes**, so the additional persisted V0 catalog/state/one-query proof in this experiment is **7,409 bytes**. Do not conflate scopes: the earlier kernel manifest and this V0 segment manifest are different records, and retained diagnostics differ.

**Excluded but observed:** construction cache/journal 866,648 bytes; held-out source corpus; temporary workspace. Required Python verifier/decoder sources, library/runtime environment and self-contained portable packaging were **not** included. Consequently 0.3360× is NOT the full independently portable GLYPH proof-bundle ratio and not a product-ready 1 TB claim.

## Evidence and explicit boundaries

- The replay worked without access to the original source path and without AI.
- The original V0 verifier reconstructs the **entire segment** in temporary storage to validate one object. This is an observable verification cost and an optimization research target; it does not negate exactness in this scoped run.
- One selected object's receipt is not proof that the full Vault candidate list was complete. No authenticated global nonmembership, currentness/rollback protection, root-signature trust anchor or committed suffix-order adjacency has been established.
- Not yet an O1–O6 complete self-contained portable evidence bundle. The verifier, codec code, and Python environment were externally available from the checked-out repository.
- No start/open RSS, cold-start latency distribution, build peak RSS, or scalable proof-size curve was measured by this specific probe.
- The numerical ratio is corpus-bound and does not transfer to enwik8, personal files or 1 TB.
- The standalone prior synthetic G1A gate confirmed hostile mutations across receipt, index, object map, manifests and root: https://github.com/yasha1971-coder/glyph-engine/actions/runs/42083532832 (29/29).
- The historical V0 proof-carrying Canterbury gate is separate; passing this 30-object public probe does not silently promote its scope.

## Next falsifiable gate

**CORE-G1C — portable independently replayable proof package + selective verification cost.**
Start from the existing V0 producer/verifier and the established O1–O6 portable evidence semantics. Make a *disposable, explicitly versioned experimental bundle*, do not modify frozen V1 or the current proof receipt format. A fresh verification process with no source repo/network must (a) bind an externally pinned corpus root and query, (b) verify at least one selected presence result, (c) reject incomplete or corrupt manifests and stale root, (d) restore correct object bytes, and (e) report verifier package bytes, peak RSS and **bytes decoded** for each proof. The first version may report full-segment decode as a limitation. Only after this passes should we test authenticated bounded extraction instead.

A global absence/count cryptographic proof and authenticated suffix adjacency are **separate, higher-level research gates**. They cannot be inferred from a valid selected-object presence receipt.

Freeze/release verdict: **NO FULL CORE FREEZE**; preserve G1A/G1B experimental receipts and leave all owner-controlled merges pending.

## Publication correction

An earlier successful CI run 42084807407 accidentally uploaded the complete disposable public fixture in a temporary Actions artifact (315 files). No private source was involved. The workflow was then corrected to upload **only result.json and command_log.json**. The authoritative reproducibility record above is run 42085257299, whose resulting artifact has **two files** (2,936 compressed bytes) with a 30-day retention. No historical workflow artifact was deleted.
