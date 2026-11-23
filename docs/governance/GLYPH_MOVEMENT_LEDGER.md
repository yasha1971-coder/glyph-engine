# GLYPH movement ledger

## 2026-11-23 — Preserve historical I0 result and select current source coverage

Source checkpoint: `390c89747157fe3c01e7f080615beb6cbacc31fc` on
`codex/core-l10-append-replay-20261106` (Draft PR #29).

The original `benchmarks/results/GLYPH_EMBEDDED_I0_CONTRACT_V1.json` remains
unchanged, SHA-256
`3f7cce1c03a4c409163f58131f77b1b04e9efd981fa4d3797acea82a45ead51c`.
Its recorded scan covers 38 tracked implementation source files. The current
unchanged checker covers 43 and reports no Embedded ABI implementation hits.
The Personal Memory `gear_native.cpp` already explained the first 38-to-39
increase; subsequent native research sources bring this checkpoint to 43.

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
