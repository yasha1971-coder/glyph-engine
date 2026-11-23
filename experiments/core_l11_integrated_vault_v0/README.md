# CORE-L11: indexed append identity bridge — bounded integration research

Status: CANDIDATE until independent CI confirms. Same Draft PR #29 and existing Issue #28; no merge.

## Industrial architectural choice

Reuse immutable content+snapshots pattern of restic/Borg; search real physical index leaves as Tantivy searches immutable segments; reserve RFC9162-style consistency plus signed/checkpointed currentness as future gates. This bridge implements neither Borg's FastCDC nor Tantivy's token index nor RFC9162 signed consistency. All current SHA pins are unsigned.

## Exact implemented contract

Original L10 physical A0/A1 roots contain 9 then 12 real physical RLB2/RLR2/EF leaf slots. Original source bytes are NOT retained. Separate domain-separated CORE-L11 logical roots bind doc_id, revision, path, source_SHA and physical leaf. Three research snapshots:

V0: nine logical docs, doc1 rev0 uses leaf1 with alpha LEFT. ONLY_IN_V1 is absent.
V1: doc1 rev1/renamed uses new leaf9 with ONLY_IN_V1 shared, plus new doc9 on leaf10 (early-sorting pathname) and doc10 on leaf11. Old leaf1 remains retained for old snapshot, never renumbered.
V2: tombstone doc6 with no change to physical A1; visible dup count drops from 3 to 2, older snapshot remains independently verifiable.

Producer queries the REAL L10 RootView full per-leaf physical index, then projects only the visible logical revision results. It outputs full physical receipts plus logical bridge receipts in a synthetic bounded research profile.

Independent verifier on a fresh runner uses existing SHA-pinned CORE-L8 native materialization to restore all 12 leaf sources from RLB2/RLR2/EF; validates byte-perfect hashes; replays unchanged CORE-L10 independent per-leaf physical proofs for each unique root/query; independently scans the restored bytes and computes correct logical coordinates/count/empty, versioned by root. The ephemeral decoded source is never included in final retained artifacts.

## Acceptance — must be measured, never assumed

18 logical query replays; 12 unique independently replayed physical query receipts; 12/12 source leaves restored (one empty source handled explicitly); 18 hostile receipt/re-pinned root mutations rejected; original historical Composition 25/25 EXACT; no original leaf rewrite; CI all PASS. A single missing source/leaf or wrong root must be UNTRUSTED, never PROVEN_EMPTY.

## Important remaining blocks

Non-normative tiny fixture; no production arbitrary-file ingest or durable source commit, no 1GB/1TB operation, no signed/checkpointed freshness/rollback/fork protection, no compact negative cryptographic proof, no full-package 30% savings evidence. Existing frozen Composition V1 operator format is NOT byte compatible with these leaf binding manifests. Do not call an unsigned pinned SHA an authoritative current snapshot.

References: https://restic.readthedocs.io/en/v0.18.1/design.html ; https://borgbackup.readthedocs.io/en/latest/internals.html ; https://docs.rs/tantivy/latest/tantivy/ ; https://www.rfc-editor.org/rfc/rfc9162.html.

Next after PASS: real durable user ingest and atomic publish; bounded full-loop exact restore on larger corpus; independent cross-host replay and full installed package accounting. Do not divert to unrelated speed benchmarks.
