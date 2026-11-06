# CORE-L9-A/B — bounded exact-query research capsule

Implements Issue #24 and `loc_text_ef_v0/CODEX_CORE_L9_HANDOFF.md` on top of
research HEAD `e8e2d98b8f68ea87ac616358a034862d7651cfd3`. The immutable fixture,
RLB2/RLR2 formats and RankV2 implementation come from
`c33cb493bc75f13ee4a904fabb190a29a196c94b`. No prerequisite source, normative
contract, frozen V1 file, ABI or production path is changed.

The scope is **VERIFIED_SMALL**, conditional on the two named CI jobs passing,
with an **externally supplied, unsigned root**. Receipts do not establish who
authorized that root, freshness, rollback resistance, semantic truth, complete
version history or a multi-document claim. This is full replay, not a succinct
cryptographic proof or a formally verified implementation. The proposed V0
receipt profile is experimental and remains subject to owner review.

## Observed prerequisite

CORE-L8 small CI [42144432624](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42144432624)
and 100MB CI [42144629682](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42144629682)
were inspected and both were SUCCESS. The latter's native materialization and
independent replay steps actually executed; this work does **not** rerun the
100MB corpus. The unchanged small L8 gate is rerun locally and in the producer
CI job: 11 materialization and four offline-replay hostile rejections.

## Actual data and code paths

1. `fixture.py` is a construction-only process. The original pinned generator
   produces `b"\x00banana_bandana\xff\x00banana!"` (24 bytes), SA and BWT. The
   existing physical EF builder samples **text SA values** at step 2; the
   construction-only legacy LOC1 is removed. Every construction file is deleted
   before a separate query process starts. The retained query state contains
   only `root.json`, `rlb2`, `rlr2`, `ef`, `decoder`.
2. `producer.py` snapshots and SHA-checks the externally pinned root and all
   mandatory components. It uses frozen deterministic RLR2 verification,
   validates RLB2 decoded-BWT integrity (temporary BWT deleted before query),
   and adds a local EF checkpoint/padding validation adapter. It uses the
   **existing** `make_runtime` / `RankV2` backward search and full locate,
   alongside `EFByteExtractor`. Complete source bytes are reconstructed in
   transient chunks and SHA-verified **before** any query claim. There is no
   raw-source input, SA input or separately retained source archive.
3. Exact count is the FM interval size. When bounded, full locate is performed
   and sorted before the requested canonical prefix is chosen. The existing
   runtime's capped FM-interval prefix is **not** mistaken for a canonical
   corpus-offset prefix. Excerpts are extracted from EF+RLB2/RLR2 and must equal
   the query bytes. Count is distinct from returned-offset length.
4. `verifier.py` runs alone with Python `-I` from another directory. Its import
   graph contains standard-library modules only; it neither imports nor
   executes producer, FM, EF or decoder code. Root, component bytes, decoder
   identity and a separate L8 materialized source are authenticated. A bounded
   source-derived SA/BWT **oracle** independently validates every RLB2 run,
   RLR2 checkpoint and physical EF sample, including padding and checksums.
   This is independent small-state validation, not a replacement search
   runtime. A full overlapping byte scan then checks count, all sorted offsets,
   all excerpt hashes and any empty claim. The caller supplies the expected
   root, pattern and limit independently of the receipt.
5. The producer CI job uploads a bounded **index-only capsule**, with decoder,
   root, example receipts and an inventory, without raw source/SA/BWT. The
   separate hostile job downloads it, re-materializes its own SHA-checked
   oracle via unchanged L8, copies the verifier alone, and runs adversarial
   tests. The artifact is replay data, not a signed authority/currentness root.

The producer checks component hashes before opening the runtime. The verifier
reads each bounded input into an immutable byte snapshot before checking it.
As in L8, a privileged adversary replacing trusted code or filesystem ancestors
concurrently is outside this local research threat model. Python, standard
library, OS, compiler and git are explicit trusted external dependencies.

## Proposed receipt serialization and bounds

`GLYPH_CORE_L9_EXACT_QUERY_RECEIPT_V0` uses ASCII JSON with sorted keys, compact
separators, `ensure_ascii=True`, no nonfinite numbers, and exactly one trailing
newline. Duplicate/unknown keys, noncanonical encodings, invalid UTF-8,
truncation, bool-as-int identities, wrong lengths and policy mismatches fail.
There is no implicit schema migration and no document ID synthesized from an
offset. Offsets always address this single complete corpus.

The record binds the external root SHA, source length/SHA, all three physical
component lengths/SHAs, decoder binary SHA, exact runtime/verifier **content
revision**, scope, query hex/length, count, full/limited policy, ascending
corpus offsets, excerpt lengths/SHAs/byte-inclusion flags, status and reason.
Content revision is SHA256 of the canonical named code-identity map, rather
than a self-referential Git commit field. CI/result metadata separately names
the exact executing Git commit. Excerpt bytes are not included.

Limits: source 1..4096 bytes; each component and decoder <=1 MiB; root <=8192
bytes; nonempty pattern <=4097 bytes; result limit 0..256 hits; receipt <=65536
bytes. Count >256 preserves exact FM count but does not enumerate offsets and
returns `UNTRUSTED / LIMITED_COMPLETENESS`. Otherwise a limit below count
returns a canonical prefix and the same untrusted completeness status. The
independent verifier accepts **only complete** receipts, with exit 0 and
`VERIFIED_SMALL`; limited, invalid or missing evidence yields exit 3 and
`UNTRUSTED`. CLI parse errors use argparse's nonzero exit 2. An empty exact
interval can be complete even with limit 0, after all mandatory checks.

`FOUND` means byte-exact matches at every reported corpus offset with complete
coverage. `PROVEN_EMPTY` means zero exact byte matches in the single specified
24-byte source under the caller-trusted root. Accept a query evidence claim
only after independent verifier success; a producer status alone is not that
independent replay. Other roots/versions/corpora are outside the statement.

## Executable coverage

- All 300 nonempty original source substrings, including duplicate pattern
  values as separate cases, plus 16 binary/boundary/absent/case/space queries.
- 316 count/full-offset checks, 532 explicit physical excerpt checks, including
  353 crossing sample boundaries; 316 deterministic receipt comparisons and
  standalone verifier replays. Empty query and empty source roots rejected.
- Limits 0/1/2 retain `count=9` for `a`, return limited evidence, exit 3 on
  independent replay. A supplemental real 257-byte repetitive index checks
  the enumeration budget: `aa` => 256 full hits/excerpts independently replayed;
  `a` => count 257, offsets [], UNTRUSTED. It uses the same frozen encoder and
  rank builder and does not replace the required 24-byte fixture.
- 59 named verifier hostiles, all exit 3/UNTRUSTED; 22 producer negatives.
  Includes three separate forged-empty mechanisms, missing rank/EF, changed
  count/offsets/query/root/source/excerpts/policy/scope/runtime, malformed JSON,
  and **re-pinned** bad EF/rank geometries, EF padding/checkpoints, a forged
  rank counter with recomputed checksums, duplicate/unknown root fields and
  zero source length. Every case/reason/exit appears in `hostile-result.json`.
- Two valid independent controls and a producer run with the oracle withheld.
  The verifier import graph is checked by AST and executed as a lone copied
  file; it does not share a serialization module with the producer.

## Reproduction

Use a clean disposable copy of this branch, a separate detached checkout at
the frozen reference above, and an empty scratch directory. No pip packages
are required. Activate the existing environment when using the GLYPH workspace.

```bash
source /workspace/glyph-env/activate.sh
g++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread \
  experiments/loc_text_ef_v0/native_run_stream_inverse_rlb2_ef.cpp \
  -o "$SCRATCH/l8-decoder"
python3 -I experiments/core_l9_exact_query_v0/run_gate.py --phase exact \
  --reference "$REFERENCE" --decoder "$SCRATCH/l8-decoder" \
  --output "$SCRATCH/exact"
python3 -I experiments/core_l9_exact_query_v0/run_gate.py --phase hostile \
  --reference "$REFERENCE" --capsule "$SCRATCH/exact/capsule" \
  --output "$SCRATCH/independent"
```

Expected final markers: `CORE_L9_A_B_EXACT_SMALL_PASS` and
`CORE_L9_INDEPENDENT_HOSTILE_REPLAY_PASS`, both orchestration commands exit 0.
The independent negative commands intentionally exit 3. The two-job workflow
is `.github/workflows/core-l9-exact-query-small.yml`. CI runs the exact PR head,
not a synthetic merge commit, and pins frozen reference by full SHA.

## Examples and resource accounting

`examples/found.json` and `examples/empty.json` are full canonical receipts from
the local compiler/environment, with `examples/root.json`. The source SHA is
`6194c5aa70291a322d8ef1e64e338958523cf7d881664d7c3b6ebbee9a6c4bb1`.
Their external test root is
`fb3753c130869776e59a3c4e762b0da7a5f138c6094b8e4f0574efd952214a84`:

- `62616e616e61` (`banana`): FOUND, count 2, offsets `[1,17]`, six-byte excerpts
  SHA256 `b493d48364afe44d11c0165cf470a4164d1e2609911ef998be868d46ade3de4e`.
- `616273656e74` (`absent`): PROVEN_EMPTY, count 0, offsets/excerpts `[]`, full
  single-corpus coverage. This says nothing about another file or generation.

Decoder binary hashes/root vary with compiler/build environment. Each CI
capsule carries its own exact pins and receipts; do not replace the external
pin of an existing receipt just to make a new build pass.

`exact-result.json` records executable-code identities and sizes, all retained
RLB2+RLR2+EF bytes, root and decoder bytes, construction/gate/native-build source
sizes, receipt bounds, temporary snapshots/rebuilt-rank/validation-BWT bytes,
fixture disk usage, RAM, CPU and wall time. It separately reports the 24-byte
verifier-only oracle, the ephemeral 52-byte sample map, and the 257-byte budget
fixture's full-verification work. The 316 primary independent replays read
7584 source bytes; positive-replay SHA operation/byte counters include code and
index hashing. Negative-suite wall/RSS and per-case exits are reported
separately; its failed-prefix hash/I/O work is not included in positive totals.

Local index sizes: RLB2 187 + RLR2 8576 + EF 96 = **8859 bytes**, root 608 and
decoder 88040 bytes. No compression advantage is asserted for a 24-byte case.
The index-only uploaded capsule is capped at 1 MiB. Code support is accounted
separately from that upload, not treated as zero-cost. File lengths are not
filesystem allocated blocks or sampled peak disk usage; the reported private
snapshot and transient validation bounds must also be considered. Python/OS/
stdlib/compiler installations are not vendored or included in a total installed
footprint claim. No retained second corpus payload is part of search state.

## FALSIFIERS / FAILED_PREDICTIONS

Preserved local sequence, before publication:

1. Initial exact gate: exit 0, 316 queries / 532 excerpt checks. Prediction
   that mandatory hash verification plus the existing EF constructor/extractor
   checked all required state was **false**.
2. First hostile gate: exit 1. Re-pinning an EF with its only select checkpoint
   changed from 0 to 1 allowed the initial L9 producer to emit PROVEN_EMPTY for
   `absent` (producer exit 0). The independent source-derived physical verifier
   rejected it. Failing root:
   `b4bf43faec5dd6706af9ff9d26ee5e2def1776fde5a21227fdf69c57f4b85411`.
   Full source reconstruction uses high bits/quotients, whereas an empty FM
   interval never calls locate/select: source SHA alone could not detect this
   invalid checkpoint.
3. Repaired **only the new L9 adapter**: validate every saved EF checkpoint and
   unused packed padding before any query. Reuse frozen full RLB2 validation
   and deterministic RLR2 rebuilding. Existing EF/runtime files stay unchanged.
4. Exact regression passed again; 53 initial hostiles passed. Added stronger
   tests, then 59 verifier hostiles and 22 producer negatives passed, including
   the original failing case. Two-job capsule transfer is tested locally too.

The original failed run is not represented as a historical PASS. This repair
does not prove the prior runtime globally correct. Any future wrong byte,
offset, count, forged empty acceptance, unaccounted retained payload, changed
frozen source or suppressed hostile failure falsifies the corresponding claim.

## Remaining owner decisions and limits

No L9-C document mapping/composition proof, O3/O4 closure, signatures,
freshness/currentness witness, rollback/fork prevention, 64-bit/TB validation,
C ABI freeze or production integration. The bounded independent source-derived
SA oracle is deliberately quadratic and cannot justify 100MB scaling. Existing
six packed-backend SKIPs and missing bounded-memory RLB3X patch lineage remain
open predecessor obligations; this gate neither executes nor closes them.
CORE-L8's historical 100MB replay is not L9 query evidence. No Move-r/Orbit
comparison, novelty claim, Lean implementation proof or universal size ratio.
The original spec's later-gate letters differ from Issue #24's C/D/E roadmap;
this PR covers only A/B and assigns no new meaning to those later gates.

Owner review is required before changing evidence semantics, multi-document
coordinates or escalating to a 100MB query candidate. No merge/rebase/tag,
release, main change, remote-host action or user-data ingest accompanies this PR.
