# CORE-L9-S — bounded-memory full-source exact-query replay

Research implementation of [Issue #26](https://github.com/yasha1971-coder/glyph-engine/issues/26),
stacked on PR #25 HEAD `7a48c499a0fc601cc6289e5644e3b2db986e7aba`.
Only this new directory and its own workflow are added. PR #25's small
quadratic verifier/caps/receipt examples, PR #21, frozen V1, normative contracts,
main and production remain unchanged. No merge authorization.

The target verdict is **VERIFIED_100MB_SCAN** only when both actual 100MB CI
jobs complete successfully. Small synthetic runs emit **VERIFIED_SMALL_SCAN**,
never the 100MB verdict. Before CI, only the explicitly reported local small
and streaming tests are evidence; no 100MB performance is predicted as measured.

## Evidence boundary

This independently replays an exact byte query over one complete SHA-identified
corpus. It is **O(n) full-source verification, not a succinct authenticated
FM-index proof**. Index checks independently establish structural constraints
and component SHA identities; they do **not** independently establish every
rank prefix/LF relation or EF suffix value against source at scale. That limit
is explicit in every verifier result (`independent_index_source_semantics=false`).
The exact answer is justified by the independent complete source scan.

The external root is unsigned. Neither this protocol nor its receipts establish
source authorship, currentness, rollback/fork resistance, all history, O3/O4,
canonical multi-document coordinates, a frozen C ABI, 64-bit/TB admission or a
production P/R/O service. Caller-supplied root/query/policy and trusted verifier
code are assumptions. A privileged concurrent attacker changing trusted code or
filesystem ancestors is outside the local research threat model. Source input
is held by a regular FD, checked for metadata changes and rehashed during scan.

## Prerequisite inspection and immutable pins

At task start PR #25 was still Draft/open/unmerged at the HEAD above. Its four
CI conclusions are SUCCESS (specialized PR 42150610413 / push 42150563032;
regular PR 42150610426 / push 42150563063). Existing README and actual source
were inspected. CORE-L8 100MB
[42144629682](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42144629682)
was independently inspected: workflow HEAD
`bbcaa5c141c84b0dc14587c24b78d141e83555fc`, job `127653008796`, SUCCESS,
native before-publication materialization and standalone byte replay actually
executed. Historical root was
`c1fbcf2d9aa8014b9f3ee2549387894185ce7192cec8ab7d889f95629b0b7e5b`;
it is not reused for a differently compiled decoder.

Original frozen tools: `c33cb493bc75f13ee4a904fabb190a29a196c94b`.
Official source: `https://mattmahoney.net/dc/enwik8.zip` (ZIP 36,445,475 bytes).

| Asset | Bytes | SHA-256 |
|---|---:|---|
| enwik8 | 100000000 | `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8` |
| canonical SA | 400000064 | `a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727` |
| canonical BWT | 200000058 | `d00b44f165ba78dc11bdc048f8578021003f17fa19587fea944e499fb6514195` |
| RLB2 | 73365105 | `4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3` |
| RLR2 | 12844028 | `e76a49db554860bf907aa307a503240d1e71ef0400282c34548a9726a3f22719` |
| physical EF | 2844324 | `c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7` |

All are checked by the unchanged L8 construction gate before it deletes
source/SA/BWT and before a root is generated/used. Production-profile L9-S
additionally requires these source/index pins; it fails if they differ.
Re-pinned malformed EF/rank inputs are structurally rejected **before** that
historical-identity check, preserving the original checkpoint regression.

## Actual implementation

- `producer.py` is an isolated adapter, reusing frozen RankV2/QueryRuntime,
  research physical EF and `EFByteExtractor`. Every mandatory component and
  decoder is privately copied and SHA-verified. Rank checkpoint dimensions,
  monotonic prefix geometry, all EF coverage, select checkpoints and padding
  are validated before a query, including an empty interval.
- Full source authentication uses unchanged L8/C++ streaming materialization.
  The temporary source is SHA-verified by L8, then **deleted before search**.
  No source path is accepted by `query()`. Python does not repeatedly reconstruct
  the entire source through EF. Count/full locate and per-hit excerpt recovery
  use only retained RLB2/RLR2/EF.
- `produce_batch.py` runs separately from the test oracle process. The test
  oracle is withheld while producer executes. It receives only physical-state
  paths and bounded queries. Startup/source authentication happens once per
  batch; deterministic queries are executed twice and measured separately from
  startup. Native materialization can have substantial run-state RAM; the
  standalone verification memory bound does not describe that C++ backend.
- `verifier.py` is a **standalone stdlib-only file**. No producer/FM/EF/L8 imports,
  no native decoder invocation, no source-derived SA/BWT, no full-source mmap.
  It hashes the root, mandatory components, decoder and named runtime/verifier
  code. It independently checks RLB2/RLR2 header/size/binding/payload SHA,
  every rank checkpoint's structural prefix geometry, all EF samples/select
  checkpoints/quotient coverage/padding. It does not claim full index semantics.
- Source pass 1 authenticates SHA/length before searching. Pass 2 scans the
  **same FD** to EOF while hashing again, checking the final SHA and file
  signature. Buffers are 1..4 MiB, carry is `pattern_bytes-1` (<=4096). Matches
  overlap; chunk-crossing hits are counted exactly once. Only the first 256
  offsets/excerpts are held; occurrence count continues to EOF without a cap.
- Both jobs use the exact PR head and frozen reference SHA. Job 1 rebuilds from
  the official ZIP, checks every historical identity through unchanged L8,
  deletes construction inputs, and produces an index-only capsule <=96 MiB.
  Job 2 verifies its inventory and the root transported **outside** the capsule
  through a GitHub job output, re-materializes its own ephemeral source, copies
  the verifier alone and replays every query and hostile. No raw source/SA/BWT
  is uploaded. Artifact hashes/root are test transport, not signed authority.

## Receipt semantics and exits

The original `GLYPH_CORE_L9_EXACT_QUERY_RECEIPT_V0` serialization and meanings
are preserved: canonical sorted compact ASCII JSON, trailing newline, exact
keys/types, external root/source/index/decoder/runtime identities, exact query,
full/limited policy, ascending corpus byte offsets and physical excerpt hashes.
Only the runtime content profile changes to `CORE_L9S_STREAM_SCAN_V0`, binding
the new adapter/verifier revision. This is a research profile, not a normative
ABI or a silent upgrade of the old small verifier's proof strength.

FOUND and PROVEN_EMPTY require complete independent source replay, all mandatory
identities/structure and exact count/offset/excerpt agreement. Empty is scoped
to one complete committed source, with no fabricated document coordinate.
Count >256 returns `UNTRUSTED / LIMITED_COMPLETENESS`, empty offsets and retained
exact count. A smaller limit at count<=256 returns a canonical prefix and the
same untrusted completeness status. Limited results still receive an independent
full-source count check; the verifier records `count_verified=true` only after
EOF and source SHA checks, but intentionally exits **3** for incomplete evidence.
Full valid replay exits **0**. All invalid evidence exits **3 / UNTRUSTED**;
argparse syntax errors exit 2. A missing/truncated/changing dependency never
becomes PROVEN_EMPTY.

Caps: source <=100000000; each index <=128 MiB; EF structural buffer <=4 MiB;
EF sample bitmap <=1000000 samples; decoder <=1 MiB; root <=8192; nonempty
query <=4097 bytes; <=256 enumerated hits; receipt <=65536 bytes. Synthetic
`--test-fixture` opt-in is limited to <=4096 bytes and emits only the small
verdict. Production mode accepts only canonical enwik8 pins. Each independent
verified query is required to stay <=64 MiB measured process RSS in CI.

## Tests and measurements

`stream_tests.py` runs 20 direct scanner cases at real 1 MiB and 4 MiB buffer
sizes, with deliberate binary chunk-crossing matches, repeated overlapping
one/two/three-byte queries, high-count truncation and five source-mutation/
signature checks. The deterministic synthetic boundary source is 2,097,168
bytes and is removed after the test. Small fixture replay uses the unchanged
24-byte binary fixture and six query cases, including limited output.

`replay_gate.py` copies and runs the verifier alone under Python `-I`, inspects
its import graph and executes **38 named hostile cases**: false empty against a
positive, query substitution, truncated-prefix empty claim, final source byte
mutation, wrong root/source, wrong mandatory RLR2 digest/size, missing
dependencies, off-by-one count, duplicate/missing/unsorted offsets, fake excerpt,
wrong query/policy, malformed/duplicate/nonfinite/oversize JSON and re-pinned
EF checkpoint/padding plus rank geometry. The three last producer geometry
mutations must also fail before any zero-hit claim. Per-case exits/reasons and
failed-prefix hash/source/component work are retained in `replay/result.json`.

The full canonical run chooses **seven** queries reproducibly: two rare
multi-hit phrases (each 2..16 hits), verified absent binary phrase, phrase at
source offset 99999937, a present non-ASCII phrase, a phrase crossing the
1 MiB scan boundary, and common `e` with >256 hits. Test-oracle selection is
separate from producer and its extra source reads/candidate counts are disclosed.
No guessed performance is substituted for actual measurements.

Each query report includes actual producer count/locate/excerpt wall/CPU/RSS,
independent scan wall/CPU/RSS, full scan bytes and **two source read passes**,
SHA operations/bytes, count and offsets. Result metadata records exact HEAD,
source/index/root/decoder identities, all runtime code identities/sizes, support
source bytes, query receipt sizes, scratch file lengths and ephemeral oracle
size. `commands.json` records commands, exits and wall times. Capsule inventory
records exact retained payload bytes. CI artifacts have finite retention:
capsule 14 days, results 30 days; replay code remains versioned.

The retained three-file index subtotal **89,053,457 bytes is not a complete
authenticated-package ratio**. Add manifest, decoder, runtime/verifier code,
receipts and metadata; tooling/OS/Python/stdlib/compiler installations are
external trusted dependencies, not zero-cost or included in an installed-size
claim. Native C++ build support and test/measurement support are separately
counted. Source/SA/BWT construction alone stages 700,000,122 bytes, before
indexes; ZIP stages 36,445,475 bytes and is deleted before SA construction.
Each producer startup privately snapshots the three components+decoder once,
and L8 snapshots them **again**, writes an ephemeral 100MB source and rehashes
it before deletion. The test oracle separately materializes 100MB. Source
positive replays read 200MB/query, not 100MB. Negative prefix work, repeated
component hashing and test selection reads are reported separately. File-length
disk measurements are not sampled peak allocated blocks; snapshots, native
output and downloaded/exported capsule copies must be added to staging costs.

Predicted complexity for a fixed bounded pattern: O(n) source I/O, O(occurrences)
match enumeration, O(chunk + pattern + 256*pattern) scan storage, independent of
source length. Structural index verification adds O(index bytes + EF samples +
257*rank checkpoints). These predictions are not formal implementation proofs;
the 100MB measurement must actually pass before the 100MB verdict is reported.
Native source authentication/query costs and run-state RAM are separate.

## Reproduction and attempt history

From a disposable checkout of this branch and separate frozen reference:

```bash
source /workspace/glyph-env/activate.sh
cmake -S "$REFERENCE" -B "$REFERENCE/build"
cmake --build "$REFERENCE/build" --target build_sa_binary_v1 build_bwt_binary_v1 -j2
g++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread \
  experiments/loc_text_ef_v0/native_run_stream_inverse_rlb2_ef.cpp -o "$SCRATCH/l8-decoder"
python3 -I experiments/core_l9_scale_v0/stream_tests.py --output "$SCRATCH/stream"
python3 -I experiments/core_l9_scale_v0/gate.py --phase small \
  --reference "$REFERENCE" --decoder "$SCRATCH/l8-decoder" --output "$SCRATCH/small"
python3 -I experiments/core_l9_scale_v0/gate.py --phase produce \
  --reference "$REFERENCE" --decoder "$SCRATCH/l8-decoder" --output "$SCRATCH/produce"
# EXPECTED_ROOT is transported separately from the capsule by the caller/CI.
python3 -I experiments/core_l9_scale_v0/gate.py --phase replay \
  --reference "$REFERENCE" --capsule "$SCRATCH/produce/capsule" \
  --expected-root "$EXPECTED_ROOT" --output "$SCRATCH/independent"
```

The official source host is absent from the local managed environment's allowed
network policy. No proxy bypass or alternate unpinned dataset was used. Local
tests execute scanner/small hostile gates; the official full rebuild executes
in GitHub-hosted CI. Existing CMake/xxd/g++ were inspected; no local dependency
installation or environment replacement was performed.

**FALSIFIERS / FAILED_PREDICTIONS:** the original L9-A/B select-checkpoint
failure remains preserved in its original README and is replayed with a new
root pin here. Initial L9-S local scanner (20 cases) and small gate (37 hostiles)
passed. At this documentation revision no executed L9-S semantic test failure
has been concealed or relabeled PASS. Full CI outcomes must be linked in the
Draft PR/final report; a pending/failed full run is not VERIFIED_100MB_SCAN.
Wrong counts/offsets/excerpts, false empty, wrong historical SHA, unbounded
memory, hidden retained source or skipped required jobs falsify the verdict.
The stronger small verifier's SA semantic oracle is not predicted to scale;
this standalone source scan intentionally makes the narrower declared claim.

The old six packed-backend SKIPs, missing bounded-memory RLB3X patch lineage,
P/R/O root authority, Composition, 64-bit/TB and deployment limitations remain
open. No merge, frozen release/tag/CONTROL change, OVH, laptop, user-data ingest
or production operation is part of this gate.
