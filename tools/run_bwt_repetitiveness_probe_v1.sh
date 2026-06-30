#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="benchmarks/results/BWT_REPETITIVENESS_PROBE_V1.jsonl"
REPORT="benchmarks/results/BWT_REPETITIVENESS_PROBE_V1.md"

TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

find \
  examples \
  benchmarks \
  corpora \
  out \
  data \
  -type f \
  -name 'bwt.bin' \
  2>/dev/null \
  | sort -u > "$TMP_LIST" || true

COUNT="$(wc -l < "$TMP_LIST" | tr -d ' ')"

if [ "$COUNT" = "0" ]; then
  cat > "$REPORT" <<'EOF'
# BWT_REPETITIVENESS_PROBE_V1

Status: no BWT files found  
Date: 2026-06-28

## Result

No `bwt.bin` files were found in the standard local paths:

- `examples`
- `benchmarks`
- `corpora`
- `out`
- `data`

## Meaning

The run-count probe is installed, but no built BWT artifacts were available to measure.

## Next step

Build or locate a GLYPH index, then run:

    python3 tools/measure_bwt_runs_v1.py path/to/bwt.bin

For the RLBWT direction, the important number is:

    r/n = BWT run count / BWT length

Lower is better for RLBWT.
EOF

  echo "no bwt.bin files found"
  echo "report: $REPORT"
  exit 0
fi

python3 tools/measure_bwt_runs_v1.py $(cat "$TMP_LIST") --jsonl-out "$OUT"

python3 - <<'PY'
import json
from pathlib import Path

jsonl = Path("benchmarks/results/BWT_REPETITIVENESS_PROBE_V1.jsonl")
report = Path("benchmarks/results/BWT_REPETITIVENESS_PROBE_V1.md")

rows = []
for line in jsonl.read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))

ok = [r for r in rows if r.get("ok")]

def verdict(r):
    x = r["r_over_n"]
    if x <= 0.001:
        return "extremely strong RLBWT signal"
    if x <= 0.01:
        return "strong RLBWT signal"
    if x <= 0.05:
        return "possible RLBWT signal"
    if x <= 0.15:
        return "weak/moderate RLBWT signal"
    return "poor RLBWT signal"

lines = []
lines.append("# BWT_REPETITIVENESS_PROBE_V1")
lines.append("")
lines.append("Status: measured")
lines.append("Date: 2026-06-28")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("Measure BWT run count `r` and `r/n` on available GLYPH BWT artifacts.")
lines.append("")
lines.append("This is the first gate for deciding whether RLBWT/r-index direction is worth engineering.")
lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append("- `n` = BWT length in bytes")
lines.append("- `r` = number of equal-byte runs in BWT")
lines.append("- `r/n` = repetitiveness signal")
lines.append("- lower `r/n` means stronger RLBWT potential")
lines.append("")
lines.append("Thresholds used:")
lines.append("")
lines.append("- `r/n <= 0.001`: extremely repetitive")
lines.append("- `r/n <= 0.01`: highly repetitive")
lines.append("- `r/n <= 0.05`: repetitive")
lines.append("- `r/n <= 0.15`: moderately repetitive")
lines.append("- `r/n > 0.15`: poor run-compression signal")
lines.append("")
lines.append("## Results")
lines.append("")
lines.append("| BWT | bytes n | runs r | r/n | avg run len | classification | verdict |")
lines.append("|---|---:|---:|---:|---:|---|---|")

for r in ok:
    lines.append(
        f"| `{r['path']}` | {r['bytes']} | {r['runs']} | "
        f"{r['r_over_n']:.8f} | {r['avg_run_len']:.3f} | "
        f"{r['classification']} | {verdict(r)} |"
    )

lines.append("")
lines.append("## Decision rule")
lines.append("")
lines.append("If real log / backup / versioned / medical / legal corpora show low `r/n`, GLYPH should continue toward compressed RLBWT runtime.")
lines.append("")
lines.append("If `r/n` is high on real corpora, do not sell RLBWT as a memory solution yet.")
lines.append("")
lines.append("## Non-claims")
lines.append("")
lines.append("- This does not prove a compact production RLBWT index.")
lines.append("- This does not prove deduplication usefulness.")
lines.append("- This only measures whether the BWT has enough runs to justify further RLBWT work.")
lines.append("- Simple exact block deduplication may still be better solved by hashing.")
lines.append("")
lines.append("## Next external question")
lines.append("")
lines.append("Ask a person working with repetitive corpora:")
lines.append("")
lines.append("> If a fixed corpus has low BWT run count and supports replayable exact-byte evidence, is that useful in your logs/backups/research workflow?")
lines.append("")

report.write_text("\n".join(lines), encoding="utf-8")
print(f"report: {report}")
PY
