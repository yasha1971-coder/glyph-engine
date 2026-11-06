# CORE-L8 — independently pinned byte identity BEFORE output publication

2026-11-06, experimental research only. Historical source and binary-safe formats frozen at \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`. [Draft PR #21](https://github.com/yasha1971-coder/glyph-engine/pull/21) only; no merge, main, OVH, public service, ABI or frozen V1 mutation.

## Original global GLYPH claim and the defect this gate targets

Goal: **compact verified searchable compression**, not just a fast BWT/LF codec. A faithful system must refuse to materialize unverified bytes as authoritative, and must bind restored bytes to committed source identity, with independent replay and explicit trust source.

CORE-L7 native \`native_run_stream_inverse_rlb2_ef.cpp\` validates internal header/FNV and EF geometry, and can produce an exact source stream, but its file candidate exists before an independently pinned **cryptographic** output check. FNV is NOT cryptographic authentication. In CORE-L8 we **do not redefine the compiled native core**; instead an additive Python stdlib gate \`trusted_materialize_v0.py\` wraps it.

## Explicit trust contract (under test, not silently elevated to production)

Input:
- external exact \`--root-sha256\` of one bounded JSON root manifest;
- the manifest must bind frozen commit, source SHA256+length, RLB2/RLR2/EF SHA256+length and native decoder SHA256;
- mandatory RLR2 is authenticated even though source inversion uses only RLB2+EF, preserving the exact-search substrate integrity;
- no raw source, SA, BWT or second compressed source payload is passed to the restorer.

Process:
1. Strict manifest parse with duplicate JSON key rejection; refuse wrong/changed external root digest and unexpected schema/fields.
2. Revalidate decoder and copy actual executable to a **private mode-0700 snapshot**; independently SHA-check *the copy to be executed* to close hash-then-exec path races.
3. Copy all retained RLB2/RLR2/EF inputs to the same private staging directory and SHA-check size and content of each copied snapshot. Any absent, truncated, changed, nonregular or mismatched component -> \`UNTRUSTED\`.
4. Launch actual, SHA-pinned C++ streaming decoder from private snapshot, with only private retained inputs. The native decoder writes a candidate under the private staging directory.
5. Independently hash the full source candidate and require exact root-bound source length and SHA. If wrong -> \`UNTRUSTED\`, no public output.
6. Only after source hash equality, publish an **existing-file-safe** hard-link at the final output path, rejecting overwrite. The returned stdout receipt states its limited evidence scope.

Two independent dependencies can be tampered independently: bytes of original source RLB2 and current rank RLR2; neither may silently be ignored. Temporary snapshots duplicate retained index bytes on disk **during verification**; they must be recorded as temporary I/O/disk costs, not counted as permanent compressed state. This adds verified failure semantics but does NOT yet provide a bound on total temporary disk or a production crash-durable root+file transaction.

**Threat boundary: IMPORTANT.** The external root digest is only as trustworthy as the caller's way of acquiring it. A user-supplied root hash is NOT a cryptographic signature, pinned independent root witness, secure timestamp, threshold authorization, rollback prevention or frozen source-of-truth update. A hostile process with privileged filesystem control is not excluded by the present prototype. Hash binding alone does NOT prove completeness or absence.

World standards for the future independent proof plane:
- [in-toto Statement v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md) uses subject artifact digest binding; GLYPH can adopt a compatible externally verifiable statement/envelope after independent semantics review, instead of claiming invented "new provenance".
- [TUF rollback/freeze specification](https://theupdateframework.github.io/specification/v1.0.36/) requires trusted version/time state and signed root/target metadata; a single SHA does not establish it.
- [Mathlib proof methodology](https://github.com/leanprover-community/mathlib4), already used for LF arithmetic, may later formalize the exact publication state machine and resource/admission inequalities. A theorem does not audit native IO.

## Gate A — small original binary-safe source: EXECUTED PASS

Original [CI 42143532241 — PASS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42143532241), repeated after private executable snapshot [CI 42143720519 — PASS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42143720519).

Original frozen fixture contains 24 bytes including NUL/FF. One positive source byte-perfect restore and **11/11 rejected hostile scenarios**:

1. Wrong external root SHA256 pin.
2. Root JSON mutated without pin update.
3. RLB2 bitflip.
4. Mandatory RLR2 bitflip despite no RLR2 native inverse API dependency.
5. EF bitflip.
6. Mandatory RLR2 missing.
7. Deliberately wrong **source SHA** in a separately re-pinned manifest: the decoder runs, yet no unverified source file is published.
8. Decoder digest mismatch.
9. Duplicate JSON manifest property.
10. Existing output cannot be overwritten.
11. Existing output symlink cannot be followed/replaced.

Receipt also verifies the root manifest SHA, backend executable SHA and published source SHA against the root-bound fixture. These tests do not cover malware inside an otherwise pinned executable, disk interruption or currentness.

## Gate A2 — independent digest replay, no decoder — PASS

[CI 42144432624 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42144432624). An additive second verifier \`independent_materialization_replay_v0.py\` has **NO import or call to the native decoder or to \`trusted_materialize_v0.py\`**. It independently rehashes the manifest, root pin, *all three* retained index components and the final published exact source; an optional success receipt is checked for exact factual consistency, NOT believed as proof of chronology.

The original 24-byte binary-safe source was verified and publicly materialized ONLY after the first gate's full SHA check. Offline verification succeeded, then rejected **four forged/mutated cases**: (i) published source bit flip, (ii) receipt false source SHA, (iii) wrong external root digest and (iv) mandatory RLR2 changed after materialization. Total materialization pipeline negative cases 11/11; independent verifier negative cases 4/4. **No signature, witness, global nonmembership or rollback/fork protection.**

## Gate B — canonical 100MB enwik8 integrated materialization: independent CI

[100MB enwik8 CI 42143898679](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42143898679) is a separately run pending experimental gate. It must:
- Reproduce all original enwik8 source (100MB), SA (400MB), BWT (200MB), RLB2 (73,365,105B) and RLR2 (12,844,028B) pinned SHA256 plus EF file (2,844,324B) SHA256.
- Generate root **in trusted CI after confirming those preexisting fixed identities**, including exact actual decoder executable hash, and log its root digest. The root is not signed or externally witnessed, so this is CI-controlled test authority, not a full trust anchor.
- Delete raw enwik8, canonical SA and BWT before restoring; run the source verifier from frozen RLB2/RLR2/EF only.
- Restore and SHA-verify all 100,000,000 bytes before final visible output, then independently rehash that final output.
- Reject wrong external root, corrupted RLR2 and wrong postdecode source SHA without publishing any final output.
- Account actual peak RSS of the **combined verifier and decoder**, elapsed time, temporary snapshots, the invariant 89,053,457B retained three-file index subtotal and the external verifier/decoder software cost.

**EXECUTED 100MB CORE-L8 RESULT:** [Initial CI 42143898679 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42143898679), code head \`3f4ea618dbeeea3b072640e4cb24faf6b56edd33\`. Reconstructed all **100,000,000 bytes enwik8**, exact canonical source SHA256 \`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8\`, after original source, 400MB SA and 200MB BWT were deleted from the runner working directory. RLB2+RLR2+EF persistent search state is exactly **89,053,457 bytes** before manifesto/proof/verifier/code; root JSON SHA256 \`c1fbcf2d9aa8014b9f3ee2549387894185ce7192cec8ab7d889f95629b0b7e5b\` was constructed in CI from independently checked canonical component hashes and the actual compiled decoder digest. The verifier private-snapshotted the mandatory index pieces and executable decoder, and independently checked entire restored source SHA **before** publishing the source. Measured wrapper wall **24.219901s**, peak RSS **372,592kB**. All three 100MB hostiles rejected: wrong supplied root, mandatory RLR2 bitflip, expected source SHA deliberately wrong despite a separately repinned manifest. This is a **research source byte-identity materialization gate**, not authenticated author/currentness P/R/O.

**100MB independent no-decoder offline replay NOW PASS:** [CI 42144629682 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42144629682), executed head \`bbcaa5c141c84b0dc14587c24b78d141e83555fc\`. The driver explicitly invoked the **separate** \`independent_materialization_replay_v0.py\` after verify-before-publish and independently rehashed the full 100MB output, frozen RLB2, mandatory RLR2, physical EF, root and decoder SHA to exact independently checked identity. The driver requires \`CORE_L8_OFFLINE_SOURCE_IDENTITY_REPLAY_PASS\` to issue its final PASS verdict. All **three** full 100MB hostile cases were also rejected. Same root SHA256 \`c1fbcf2d9aa8014b9f3ee2549387894185ce7192cec8ab7d889f95629b0b7e5b\`, same frozen source identity, retained RLB2+RLR2+EF subtotal **89,053,457B**. Combined materialization wall **23.361670s**, peak **372,584 kB**; independent replay added an extra SHA scan and should not be mistaken for the faster wall number. The new result **closes CORE-L8 pinned byte-identity + independent replay at 100MB**, but does NOT establish source authority/rollback/currentness, exact-query O3/O4 or user production.

## Next gates after CORE-L8

1. Authentic external-root witness/rollback/fork/currentness protocol, with independently verifiable signatures or an equally explicit threat model, and canonical P/R/O coordinates + query-specific evidence; never claim global PROVEN_EMPTY from source checksum alone.
2. O3/O4 test: query -> canonical document offset -> small exact byte excerpt -> source hash/manifest -> external verifier, refusing source swaps/forged currentness.
3. Hostile disk/full/truncated writes, crash recovery, concurrent readers, time-of-check to time-of-use and atomic generation snapshots. A final file+receipt pair is not atomically published by today's code.
4. Finite resource admission **before allocation**: current n/u32 and EF ~n/s temporary RAM still block TB. Snapshot temporary storage is intentionally nonzero.
5. Real corpus classes, genuine Move-r/Orbit cross-tool benchmarks and trustworthy full compressed-state ratio including code/proof.
