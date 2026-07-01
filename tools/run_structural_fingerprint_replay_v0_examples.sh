#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="benchmarks/results/structural_fingerprint_replay_v0"
mkdir -p "$OUT_DIR"

echo "=== replay structural fingerprint examples ==="

fail=0

for artifact in benchmarks/results/structural_fingerprint_v0/*.json; do
  name="$(basename "$artifact" .json)"
  out="$OUT_DIR/${name}.replay.json"

  echo "[replay] $artifact"
  if ! tools/replay_structural_fingerprint_v0.py "$artifact" --out "$out"; then
    fail=1
  fi
done

python3 - <<'PY'
import json
from pathlib import Path

replays = sorted(Path("benchmarks/results/structural_fingerprint_replay_v0").glob("*.replay.json"))
out = Path("benchmarks/results/STRUCTURAL_FINGERPRINT_REPLAY_V0_EXAMPLES.md")

lines = [
    "# STRUCTURAL_FINGERPRINT_REPLAY_V0_EXAMPLES",
    "",
    "Status: measured",
    "Date: 2026-07-01",
    "",
    "## Purpose",
    "",
    "Replay deterministic structural fingerprint artifacts and verify that they reproduce from source bytes.",
    "",
    "## Results",
    "",
    "| artifact | ok | errors |",
    "|---|---:|---|",
]

for p in replays:
    j = json.loads(p.read_text())
    lines.append(f"| `{Path(j['artifact']).name}` | {j['ok']} | `{', '.join(j['errors'])}` |")

all_ok = all(json.loads(p.read_text()).get("ok") for p in replays) if replays else False

lines += [
    "",
    "## Decision",
    "",
    f"All replay checks passed: `{all_ok}`",
    "",
    "## Non-claims",
    "",
    "- This does not predict best codec.",
    "- This only verifies deterministic reproduction of the structural fingerprint artifact.",
    "",
]

out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
PY

if [ "$fail" -ne 0 ]; then
  exit "$fail"
fi
