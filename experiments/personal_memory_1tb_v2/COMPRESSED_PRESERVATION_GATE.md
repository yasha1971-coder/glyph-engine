# Compressed preservation integration gate

Experimental first connection to the EXISTING portable V1 Phase B host.
Not a replacement for frozen V1, an installed release, or a compact full Vault.

`compressed_preservation.py` invokes `portable_runtime.run_materialization`
with a preservation backend. Default V1 commands keep the original cache path.
New entry point requires --vault, --trust-root, --phase-a, --selection, --output,
--archive, --archive-sha256; optional --pin/--pin-sha256 and
--precompressor/--precompressor-sha256. Archive/binary pins are supplied by host,
not model output. Run --help for arguments. Do not deploy on real data yet.

Existing Phase A validation, explicit human selection receipt binding, frozen
profile checks, root/segment authentication and object maps run before reading
selected payload. Backend requires the same path, byte count and SHA-256 as the
authenticated V1 object. Host rechecks returned bytes independently before staging
publication. V2 receipts have their own format and bind the archive receipt pin.
Unselected objects are not decoded. No missing-cache requirement for this backend.

This is a cross-representation identity bridge, NOT a new trusted root or global
completeness proof. Existing V1 discovery still needs the full original search
representation. Adding a compressed archive alongside it can increase total size.
No compression ratio improvement is claimed by this gate.

Six new tests cover exact selected Precomp bytes, wrong identity, changed binary,
existing-host materialization without corpus cache, changed selection/Phase A
binding without decoding, and host rejection of a wrong backend response.
The three host-sequencing tests stub the authenticated root/segment reader and
use FAKE_PRECOMP; they do not constitute an end-to-end live authentication proof
or a real Precomp security test. Frozen profile checks are real.

Commands:
```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/glyph-v1/tests -v
```
First complete run: 50/50 in 4.855 s; existing portable tests 6/6 in 0.003 s.
Intermediate test fixture failed because it referenced nonexistent runtime.PROFILE;
corrected to read PROFILE.json. No production validation weakened.

Limits: ordinary decoding uses adapter budgets; Precomp inherits the previous
experimental decoder with timeout but no hard memory/output isolation, and
XZ transformed size is not bounded. Thus hostile-input/phone deployment is blocked.
Caller-selected output must stay outside canonical Vault/trust/archive. Concurrent
hostile mutation, decoder supply chain, persisted authorization, fsync/crash
durability and whole-root atomic publication require additional gates.

Next: a synthetic real authenticated fixture (no reader stubs), cross-version
negative cases, bounded isolated Precomp worker, preservation/search root design
without a second full media representation, and full size accounting. Retain
Composition preconditions, model-independent trust and source-deletion prohibition.

## Real authenticated fixture follow-up

`tests/authenticated_fixture.py` now constructs tiny valid RLB3X, LOC2, rank and
AUTHLOC artifacts, object maps, segment/root commitments and an external pin.
The suffix oracle sorts byte suffixes with a separate lowest sentinel; it is
explicitly bounded to 4096 bytes and is not a deployable builder.

Seven additional tests invoke the frozen real reader and the real portable host
without mocking authentication. They prove exact FM count/locate against known
bytes, authenticated absence, touched-index corruption rejection, root and
object-map rejection, external-pin rejection, selected compressed-object corruption
rejection, and selected hybrid-object restoration without a corpus cache.

Scope: one segment, two synthetic files totaling 32 bytes; ordinary hybrid codecs.
Phase A remains an explicit test receipt, not output of natural-language discovery.
Human selection is created explicitly by the test. No real Precomp executable,
live laptop, multi-segment crash recovery or new global-storage ratio is proven.
This extends rather than relabels the earlier mocked-host tests. Frozen artifacts
and production default behavior remain unchanged.
