# GLYPH local memory integration gate V1

Status: experimental read-only adapter, 2026-09-12. Not a finished Personal Vault,
not a phone benchmark, not an LLM implementation. No existing V1 code changed.

## Compatibility found by reading code

| Existing component | Reuse / integration boundary |
| --- | --- |
| `experiments/personal_vault_v0/glyph_vault_cli_v1.py` | Existing immutable RLB3X segments, publication and restoration; no hybrid segment reader. |
| `experiments/personal_vault_v0/ai_glyph_vault_bridge_v0.py` | Existing exact-probe AI bridge; specifically requires RLB3X and LOC2. Retain, do not silently substitute a scan. |
| `verified_hybrid_archive.py` | Reuse archive build/restore; adapter verifies pinned receipt and stored/decoded bytes. |
| `verified_reversible_precompression_archive.py` | Read manifest and ordinary codec objects. Precomp objects explicitly uncovered until isolated worker integration. |

## Implemented

`local_memory_bridge.py`: caller supplies an archive path, independently retained
receipt SHA-256, and explicit `(receipt SHA-256, relative path)` grants. Multiple
archive versions remain distinct even if files share a content SHA-256. The host,
not an LLM-generated plan, owns the grants and pins.

Exact UTF-8 scans return bounded fragments with version, content digest, original
path, byte offset and byte length. A fragment's bytes are verified against its
object. Files absent from the grant are neither read nor disclosed in output.
Corruption raises an error and returns no accumulated evidence; host must map
exceptions to UNTRUSTED and must never stream provisional snippets.

FOUND can coexist with incomplete coverage. PROVEN_EMPTY applies only to the
explicit nonempty grant, exact UTF-8 query and successfully scanned files.
Unsupported encoding/codec and exhausted scan budget produce incomplete coverage.
This is not semantic absence and not a search of the user's entire memory.

Default budgets: 4,096 snippet bytes, 32 MiB scanned logical bytes, 8 MiB per
stored/decoded file, 8 MiB receipt, 128 MiB XZ decoder memory limit. UTF-8 output
remains valid. These are component limits, NOT a measured process-RSS or tokenizer
budget. Manifest parsing and caller-provided archive lists still need global limits.

No network, model loading, source modification or external process invocation.
Document text is untrusted data, not instructions; the output marker alone does
not solve prompt injection. A future model adapter has no write/delete tools by
default. A model token budget and output citation validation remain necessary.

## Evidence and honest scope

Final local result: 44/44 tests, 5.038 s, exit 0. Full stdout/stderr is retained
in `LOCAL_MEMORY_GATE_TEST_OUTPUT.txt`. Baseline was 32 tests; 12 were added.
Base commit: `1f8d35f5b2b7a50c15ffc1d632a8d277bc7f2538`, base tree
`f845b652df5b36dde84e5f100e7b2f3cd9f25163` (same tree as the user's
`982533144c7e8a74e709947b527ef145db80d44a` patch-applied commit).

An intermediate run failed one new test: repeated `A` bytes selected an ordinary
codec rather than Precomp, so expecting incomplete coverage was incorrect for
that fixture. Replaced the payload with seeded pseudo-random bytes to exercise
the intended Precomp branch; no production routing was changed to make it pass.

Run from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v
```

New tests build two separate archive snapshots from synthetic files, change and
rename files, query across both versions, restore both with existing decoder,
and compare old archive bytes before/after. This does NOT implement a shared
append-only root, cross-segment physical deduplication or atomic segment publish.
Other tests cover permission scope, receipt pin, tampering, symlinks, UTF-8 byte
coordinates, limited context, incomplete coverage and bounded decompression.
Precomp compatibility test uses the existing FAKE_PRECOMP fixture, not real
JPEG recompression or the production binary. Synthetic results do not remeasure
the user's VIKA corpus or repeat its inventory.

## Limits and next work

1. Persistent host-owned version/grant catalog and append-only root integration;
   test process kill before/after publication. Current tests are not crash tests.
2. Harden filesystem access against concurrent hostile mutation (current symlink
   checks are not race-proof); enforce parser/type/global memory limits.
3. Isolated Precomp extraction worker and rebuildable search sidecars, with exact
   provenance to original bytes. OCR needs separate coordinate semantics.
4. Connect existing RLB3X query backend and this adapter through explicit backend
   and coverage contracts. Keep Composition precondition before Index Forest.
5. Add a switchable local model adapter with tokenizer budget, no network fallback,
   grounded citations and Russian/Ukrainian task evaluation on actual devices.
6. Desktop UI, then iPhone client. Phone may hold a selected offline subset; do not
   imply that a terabyte corpus fits into every phone. Benchmark startup, RAM,
   energy, latency and recovery before calling it industrial-ready.

Receipt pins bind bytes only; they are not signatures or encryption. No claim of
authenticated origin, encrypted vault, full filesystem metadata preservation,
long-term independent backup or local-LLM quality follows from this gate.
