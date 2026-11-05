# CORE-L1 — LOC1 vs LOC2, original shared RLB2/RLR2 synthetic gate

Recorded: 2026-11-05.
Scope: **SMALL BINARY-SAFE RUNTIME PARITY, NOT ENWIK8 OR FULL AUTHENTICATED RELEASE**.
Executed code head: \`c3314829681896ff2c58a79800761b2e1764a91f\`.
CI SUCCESS: https://github.com/yasha1971-coder/glyph-engine/actions/runs/42093909097.
Pinned historical V2 source: \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`.

## Executed

From original V2 \`check_rlbwt_query_v2.build_fixture()\`, construct a real 24-byte source with NUL/0xFF, canonical SA and BWT, **the same** original RLB2 and RLR2 binaries, original LOC1 and original experimental LOC2 built from the canonical SA. The LOC1/LOC2 readers reuse the same \`RankV2.backward_search\`, LF and count implementation.

Twelve binary patterns, four requested caps \`-1, 0, 1, 2\`, three repetitions: **144/144 count, FM-interval, locate-offset and LF parity comparisons PASS**. Full unbounded results also equal a separate raw-byte exact oracle. For bounded results, both return the **first FM rows then sort their offsets**, which are exact source-offset subsets and **not necessarily** source-order top-k. The initial test mistakenly required source-order top-k; it failed on a valid bounded result. That *oracle error*, not the runtime implementation, was corrected and rerun. Do not hide this historical failing first run, or rewrite the API as if it promised source-order prefix.

Fixture index storage:
- LOC1: **232 bytes**
- LOC2: **84 bytes**
- Same source, RLB2/RLR2, sample geometry and 13 sampled SA rows.
- Max LF walk observed: **3**.
- LOC2 candidate digest: \`ce46eb3c60c2a419f0a4dac586623391cd1173a3686afb4f62c86be7c3ac885a\`.

A one-byte corrupted LOC2 sample is rejected by the **separately pinned SHA and exact canonical SA sample verifier**; a corrupted header is also rejected. Important: this does not demonstrate that the original explicit experimental LOC2 runtime loader itself performs those two independent authentication steps.

## Still unproven

- Full 100MB enwik8 artifact identity, ratio and latency. Separate source-hash-pinned GitHub Actions experiment \`core-enwik8-loc1-loc2-full-replay.yml\` is responsible.
- Signed or trusted external pin/currentness, immutable production LOC2 manifest profile, O1–O6 proof-carrying portability, global cryptographic nonmembership and 1TB scaling.
- Universal storage ratio: the 24-byte fixture is purposely a correctness harness, not meaningful compression evidence.
- Statistical performance superiority from same-process timing samples.

## Next decision

Require independent rebuild of historical enwik8 SA/BWT/RLB2/RLR2/LOC1 hashes first; *then* measure original LOC2 on identical bytes with the same source-oracle and reveal all extra persistent research metadata. A separate SDSL CSA reference, pinned to its own upstream commit, provides a non-GLYPH independent compressed-index comparison. Never blend their proof/trust scopes.
