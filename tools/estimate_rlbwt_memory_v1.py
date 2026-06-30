#!/usr/bin/env python3
import json
from pathlib import Path

IN = Path("benchmarks/results/NAMED_EXISTING_BWT_RN_MATRIX_V1.jsonl")
OUT_MD = Path("benchmarks/results/RLBWT_MEMORY_ESTIMATE_V1.md")
OUT_JSON = Path("benchmarks/results/RLBWT_MEMORY_ESTIMATE_V1.json")

rows = [json.loads(x) for x in IN.read_text().splitlines() if x.strip()]

estimates = []

for r in rows:
    if not r.get("ok"):
        continue

    n = r["bytes"]
    runs = r["runs"]
    rn = r["r_over_n"]

    # Naive estimates:
    # 5 bytes/run = 1 byte symbol + 4 byte run length
    # 9 bytes/run = 1 byte symbol + 8 byte run length
    # These are lower-level BWT-run storage estimates only,
    # not full rank/select/locate production index sizes.
    rle_5 = runs * 5
    rle_9 = runs * 9

    estimates.append({
        "path": r["path"],
        "bwt_bytes": n,
        "runs": runs,
        "r_over_n": rn,
        "avg_run_len": r["avg_run_len"],
        "rle_5_bytes": rle_5,
        "rle_9_bytes": rle_9,
        "rle_5_vs_bwt": rle_5 / n if n else None,
        "rle_9_vs_bwt": rle_9 / n if n else None,
    })

OUT_JSON.write_text(json.dumps({
    "ok": True,
    "estimates": estimates,
}, indent=2), encoding="utf-8")

lines = []
lines.append("# RLBWT_MEMORY_ESTIMATE_V1")
lines.append("")
lines.append("Status: measured estimate")
lines.append("Date: 2026-06-28")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("Convert measured BWT run counts into rough RLBWT storage estimates.")
lines.append("")
lines.append("This is not a production RLBWT index size. It is only a first-order estimate for run storage.")
lines.append("")
lines.append("## Estimate model")
lines.append("")
lines.append("- `5 bytes/run`: 1 byte symbol + 4 byte run length")
lines.append("- `9 bytes/run`: 1 byte symbol + 8 byte run length")
lines.append("- Does not include full rank/select/locate overhead")
lines.append("- Does not claim production memory usage")
lines.append("")
lines.append("## Results")
lines.append("")
lines.append("| BWT path | BWT bytes | runs | r/n | avg run | 5B/run | 5B/run vs BWT | 9B/run | 9B/run vs BWT |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for e in estimates:
    lines.append(
        f"| `{e['path']}` | {e['bwt_bytes']} | {e['runs']} | "
        f"{e['r_over_n']:.8f} | {e['avg_run_len']:.3f} | "
        f"{e['rle_5_bytes']} | {e['rle_5_vs_bwt']:.3f}x | "
        f"{e['rle_9_bytes']} | {e['rle_9_vs_bwt']:.3f}x |"
    )

lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append("If `5B/run vs BWT` is not far below 1.0, RLBWT is unlikely to be a major memory breakthrough for that corpus.")
lines.append("")
lines.append("For GLYPH, RLBWT should only become a priority when named real corpora show both:")
lines.append("")
lines.append("- low `r/n`, ideally `<= 0.05`")
lines.append("- estimated run storage clearly below raw BWT size")
lines.append("")
lines.append("## Current decision")
lines.append("")
lines.append("Do not build production RLBWT runtime yet. Continue using RLBWT as a measured viability branch, not as the main GLYPH claim.")
lines.append("")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("wrote", OUT_MD)
print("wrote", OUT_JSON)
