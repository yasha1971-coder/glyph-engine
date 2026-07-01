#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="benchmarks/results/structural_fingerprint_v0"
mkdir -p "$OUT_DIR"

echo "=== structural fingerprint examples ==="

if [ -f examples/mini/out/corpus.bin ]; then
  tools/glyph_structural_fingerprint_v0.py \
    examples/mini/out/corpus.bin \
    --out "$OUT_DIR/mini_structural_fingerprint_v0.json"
fi

if [ -f /tmp/silesia_check/reymont ]; then
  tools/glyph_structural_fingerprint_v0.py \
    /tmp/silesia_check/reymont \
    --out "$OUT_DIR/reymont_structural_fingerprint_v0.json"
fi

if [ -f /tmp/silesia_check/webster ]; then
  tools/glyph_structural_fingerprint_v0.py \
    /tmp/silesia_check/webster \
    --out "$OUT_DIR/webster_structural_fingerprint_v0.json"
fi

python3 - <<'PY'
import json
from pathlib import Path

out = Path("benchmarks/results/STRUCTURAL_FINGERPRINT_V0_EXAMPLES.md")
files = sorted(Path("benchmarks/results/structural_fingerprint_v0").glob("*.json"))

lines = [
    "# STRUCTURAL_FINGERPRINT_V0_EXAMPLES",
    "",
    "Status: measured",
    "Date: 2026-07-01",
    "",
    "## Purpose",
    "",
    "Example deterministic structural fingerprints.",
    "",
    "These are not codec predictions.",
    "",
    "## Results",
    "",
    "| file | bytes | entropy | alphabet | NUL | printable | anchor k12 pairs | anchor k12 median distance |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]

for f in files:
    j = json.loads(f.read_text())
    bs = j["byte_stats"]
    k12 = next((x for x in j["anchor_repeat_profiles"] if x["k"] == 12), {})
    lines.append(
        f"| `{j['source']['name']}` | {bs['bytes']} | "
        f"{bs['entropy_bits_per_byte']:.4f} | {bs['alphabet_size']} | "
        f"{bs['nul_bytes']} | {bs['printable_fraction']:.4f} | "
        f"{k12.get('pair_count', 0)} | {k12.get('median_distance')} |"
    )

lines += [
    "",
    "## Non-claims",
    "",
    "- This does not predict best codec.",
    "- This does not replace compressor trials.",
    "- This is a replayable structural measurement artifact.",
    "",
]

out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
PY
