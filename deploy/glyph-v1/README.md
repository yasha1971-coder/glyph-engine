# GLYPH V1 — Portable authenticated retrieval

GLYPH V1 combines the Personal Vault V0.1 exact-byte lifecycle with the frozen
Laptop V1 authenticated retrieval path behind one `glyph` command.

Status: **Laptop experimental scope verified; historical-query cross-host replay
verified on OVH; production is not claimed.**

## Preserved trust boundary

```text
query
  -> deterministic compilation
  -> authenticated provenance and exact evidence
  -> candidate list
  -> explicit human selection receipt
  -> authenticated object map
  -> post-selection materialization
  -> object SHA-256 verification
  -> verified bytes
```

The model is not retrieval, selection, currentness, support, or truth authority.
The V1 path does not materialize payload during Phase A. Phase B refuses missing
cache data, unauthenticated metadata, a changed Phase A receipt, a changed human
selection receipt, or an object digest mismatch.

## One release, two host layouts

The laptop and OVH run the same source and frozen component bytes. Physical Vault,
trust-sidecar, cache, and output paths are command arguments. The adapter builds an
ephemeral compatibility layout and a derived locator pin; it never edits the
canonical pin or the frozen V6.13 resolver.

Supported verified Python range: 3.10 through 3.12 on Linux. The laptop deployment
is WSL/Linux; a native Windows runtime is not claimed.

## Verify the checked-in evidence

```bash
glyph verify-profile
glyph doctor-v1 \
  --vault "$HOME/GlyphPilot/VIKA_proekt-vault-patched-test" \
  --trust-root "$HOME/GlyphPilot/VIKA-trust2-five"
```

`verify-profile` authenticates five frozen files and six Laptop/OVH evidence files.
`doctor-v1` additionally checks the 31 required external runtime paths. It does not
run retrieval or write canonical state.

## Phase A: authenticated discovery only

```bash
OUT="$HOME/GlyphPilot/GLYPH-V1-runs/run-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

glyph query \
  --vault "$HOME/GlyphPilot/VIKA_proekt-vault-patched-test" \
  --trust-root "$HOME/GlyphPilot/VIKA-trust2-five" \
  --output "$OUT" \
  --query 'your natural-language query'
```

Inspect the candidates. Phase A stops with payload untouched.

## Human selection

```bash
PHASE_A="$(find "$OUT" -maxdepth 1 -name 'GLYPH_V1_LIVE_ACCEPTANCE_PHASE_A_*.json' -print -quit)"

glyph select \
  --phase-a "$PHASE_A" \
  --candidate 1 \
  --out "$OUT/HUMAN_SELECTION.json"
```

The selection receipt is byte-bound to Phase A and records the exact selected
candidate. This command is the authority boundary; automation must not choose the
number on behalf of the human.

## Phase B: authenticated materialization

```bash
glyph materialize \
  --vault "$HOME/GlyphPilot/VIKA_proekt-vault-patched-test" \
  --trust-root "$HOME/GlyphPilot/VIKA-trust2-five" \
  --cache "$HOME/GlyphPilot/LI-V0-LAPTOP/cache/preservation-v72" \
  --phase-a "$PHASE_A" \
  --selection "$OUT/HUMAN_SELECTION.json" \
  --output "$OUT"
```

Output is forbidden inside the canonical Vault, trust sidecars, or preservation
cache. Materialized files are staged and published only after every selected object
matches its authenticated SHA-256.

## What the evidence proves

- Laptop: one unseen ACEAPEX query produced the correct single candidate without
  payload access; after human selection, two objects were materialized and verified.
- OVH: the same historical query and frozen identities replayed successfully on
  Python 3.10.12; missing preservation cache failed closed; two selected objects
  were verified.
- OVH measured five segments, 249,967,975 logical corpus bytes and 271,897,720
  search-data bytes, ratio 1.087730218. Sub-1x storage is therefore **not** claimed
  for this Vault.

The evidence does not prove perfect recall, arbitrary-language understanding,
general model-answer correctness, multi-user production service, or equivalence for
an unseen OVH query.

## Frozen versus portable code

- `frozen/`: byte-identical V1 components and canonical acceptance pin.
- `reproduction/`: recovered historical Phase A/B producer sources. These preserve
  provenance and are not the portable API.
- `portable_runtime.py`: host adapter and current V1 command implementation.
- `evidence/`: immutable Laptop and OVH receipts bound by `PROFILE.json`.

The exact Vault corpus, rank sidecars, model weights, cache, and materialized user
objects are deliberately excluded from Git.
