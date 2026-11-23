# GLYPH movement ledger

## 2026-11-23 — Preserve historical I0 result and select current source coverage

Source checkpoint: `390c89747157fe3c01e7f080615beb6cbacc31fc` on
`codex/core-l10-append-replay-20261106` (Draft PR #29).

The original `benchmarks/results/GLYPH_EMBEDDED_I0_CONTRACT_V1.json` remains
unchanged, SHA-256
`3f7cce1c03a4c409163f58131f77b1b04e9efd981fa4d3797acea82a45ead51c`.
Its recorded scan covers 38 tracked implementation source files. The current
unchanged checker covers 43 and reports no Embedded ABI implementation hits.
On this research branch, the five additions relative to frozen C33 are:

- `experiments/core_perf_research_v0/ac_native_v0.cpp`
- `experiments/loc_text_ef_v0/native_inverse_rlb2.cpp`
- `experiments/loc_text_ef_v0/native_parallel_inverse_rlb2_ef.cpp`
- `experiments/loc_text_ef_v0/native_run_interval_inverse_rlb2_ef.cpp`
- `experiments/loc_text_ef_v0/native_run_stream_inverse_rlb2_ef.cpp`

The earlier Personal Memory PR #15 38-to-39 result involved `gear_native.cpp`
on its separate lineage. That file is absent here; it is not the explanation
for this branch's 38-to-43 result.

Generated with the existing checker, without changing its algorithms:

```sh
python3 -I tools/check_embedded_i0_contract_v1.py --write \
  --result benchmarks/results/GLYPH_EMBEDDED_I0_CONTRACT_V1_SOURCE_SCAN_43.json
```

New result SHA-256:
`b2818a4b698c2fde684c0a207b45ec9b5b9b03ff764bf4388a20d281b99257a1`.
The complete JSON field diff is only
`implementation_source_files_scanned: 38 -> 43`. `verify.sh` explicitly selects
this result with `--verify --result`; this versions source-coverage evidence,
not the ABI contract. The checker's reviewed contract commit remains
`fe94eb0314cff76191a7eaba20347c234069a8f2`.

All 13 existing mutation checks, including premature Embedded implementation,
remain enabled. This movement does not implement the Embedded ABI, change any
normative specification, promote a runtime or close historical SKIPs. Frozen
V1, prior artifacts, tags, CONTROL, main, OVH and production remain unchanged.

## 2026-11-23 — Opt-in parallel scheduling of unchanged full SHA guards

Owner resumed bounded engineering; no customer/pilot acceptance is inferred.
Code experiment: `1727b245da1721b8073bd1117f9e39906e0ceb1f`, following the
separate I0 evidence reconciliation `4811406e85f1c3b9da8df3ae3b1ff5225d959540`.
`parallel_session.py` schedules the original full catalog function with two
workers; it changes no original Session, default query, parser, checker, native
decoder, receipt or normative contract. Complete SHA reads and frozen/code
checks remain before and after each query. All reads are drained before cleanup.

Predeclared lock: `search_cost/PARALLEL_GUARD_LOCK.json`. Public hypothesis <=0.8
paired guard time was met: median 0.592668 at 100MB; all 54 external-pinned
replies match. Small fixture: all 120 replies match, but median guard ratio
1.303417 is a preserved regression. No automatic source-size threshold or
default promotion follows. The original 16 session/constructor hostiles run
against the candidate, plus an actual candidate-owned worker RSS kill; the old
worker RSS assertion is additionally preserved. Seven real-read integrity and
failure-cleanup regressions were added. Separate independent replay uses the
unchanged verifier and actual candidate receipts, with pins outside its capsule.

Only scheduling changes: 466,016,288 live/private file bytes remain hashed per
public request, plus code. Guard cost is proportional to state. Temporary state
and CPU/threads are paid; no new retained Vault files or compact-proof claim.
The root README now states the known default per-query preparation cost,
measured tradeoff and verifier full-scan boundary. Existing CI gains the complete
top-level verification and candidate regression/replay steps; no checks removed.

## 2026-11-23 — Full Git ancestry for the existing I0 provenance check

Regular PR CI45288521354 passed P/R/O but exposed a distinct setup issue:
actions/checkout's default depth-one synthetic merge commit made the unchanged
checker resolve `reviewed_commit` as `ecf6e8b8c1b0d0d9a21a1b8ea3c25814e8c86481`.
That changed the computed JSON SHA256 to
`f86451937b37045f8badf2a6af7ea96fffad0e6fce3dbbb1324d3158e7882417`.
The complete field diff from the correct SOURCE_SCAN_43 result was only
`reviewed_commit`; all source/spec hashes and checks matched.

Independently reproduced the exact merge checkout at depth one: verify exits1
with that exact digest. Fetching full ancestry while keeping the same HEAD,
tree, checker and committed artifacts makes verify exit0, resolves the actual
reviewed source commit `fe94eb0314cff76191a7eaba20347c234069a8f2`, and restores
the committed SHA256 `b2818a4b698c2fde684c0a207b45ec9b5b9b03ff764bf4388a20d281b99257a1`.
The regular CI checkout now uses `fetch-depth: 0`. No result is rewritten for
a CI merge identity, no provenance/checker logic is weakened, and the failed
run is preserved. Runtime/parallel Session behavior is unchanged.
