#!/usr/bin/env bash
set -euo pipefail

cd ~/GLYPH_CPP_BACKEND

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
  if [ -f "HDFS.log" ]; then
    SOURCE="HDFS.log"
  elif [ -f "bench_1gb/HDFS_1GB.log" ]; then
    SOURCE="bench_1gb/HDFS_1GB.log"
  else
    echo "missing source. Usage:"
    echo "  ./tools/run_multiversion_rlbwt_probe_v1.sh /path/to/HDFS.log"
    exit 1
  fi
fi

WORK="benchmarks/work/multiversion_rlbwt_v1"
RESULTS="benchmarks/results"
mkdir -p "$WORK" "$RESULTS"

VERSION_MIB="${GLYPH_MV_VERSION_MIB:-64}"
VERSIONS="${GLYPH_MV_VERSIONS:-4}"
OVERLAP_SHIFT_MIB="${GLYPH_MV_OVERLAP_SHIFT_MIB:-8}"
NONOVERLAP_SHIFT_MIB="$VERSION_MIB"

BUILD_SCRIPT="tools/build_glyph_index_v1.sh"
if [ ! -x "$BUILD_SCRIPT" ]; then
  echo "missing executable $BUILD_SCRIPT"
  echo "Need canonical GLYPH builder."
  exit 1
fi

run_case() {
  local label="$1"
  local shift_mib="$2"

  local dir="$WORK/$label"
  local corpus="$dir/corpus.bin"
  local manifest="$dir/collection_manifest.json"
  local index="$dir/index"
  local runs_json="$dir/bwt_runs.json"

  rm -rf "$dir"
  mkdir -p "$dir"

  echo
  echo "=== make collection: $label ==="
  python3 tools/make_multiversion_collection_v1.py \
    --source "$SOURCE" \
    --out "$corpus" \
    --manifest "$manifest" \
    --version-mib "$VERSION_MIB" \
    --shift-mib "$shift_mib" \
    --versions "$VERSIONS" \
    --label "$label"

  echo
  echo "=== build index: $label ==="
  "$BUILD_SCRIPT" "$corpus" "$index"

  echo
  echo "=== measure BWT runs: $label ==="
  python3 tools/measure_bwt_runs_v1.py "$index/bwt.bin" | tee "$runs_json"
}

run_case "nonoverlap" "$NONOVERLAP_SHIFT_MIB"
run_case "overlap" "$OVERLAP_SHIFT_MIB"

python3 - <<'PY'
import json
from pathlib import Path

WORK = Path("benchmarks/work/multiversion_rlbwt_v1")
OUT_MD = Path("benchmarks/results/MULTIVERSION_RLBWT_PROBE_V1.md")
OUT_JSON = Path("benchmarks/results/MULTIVERSION_RLBWT_PROBE_V1.json")

cases = []
for label in ["nonoverlap", "overlap"]:
    d = WORK / label
    meta = json.loads((d / "collection_manifest.json").read_text())
    runs = json.loads((d / "bwt_runs.json").read_text())
    cases.append({
        "label": label,
        "collection_bytes": meta["out_bytes"],
        "version_mib": meta["version_mib"],
        "shift_mib": meta["shift_mib"],
        "versions": meta["versions"],
        "source": meta["source"],
        "source_sha256": meta["source_sha256"],
        "bwt_bytes": runs["bytes"],
        "runs": runs["runs"],
        "r_over_n": runs["r_over_n"],
        "avg_run_len": runs["avg_run_len"],
        "classification": runs["classification"],
        "sha256": runs["sha256"],
    })

non = next(x for x in cases if x["label"] == "nonoverlap")
ov = next(x for x in cases if x["label"] == "overlap")

improvement = non["r_over_n"] / ov["r_over_n"] if ov["r_over_n"] else None

decision = "unknown"
if ov["r_over_n"] <= 0.05 and improvement and improvement >= 2.0:
    decision = "multiversion_hypothesis_alive"
elif ov["r_over_n"] <= 0.05:
    decision = "overlap_signal_present_but_needs_real_versions"
else:
    decision = "no_strong_multiversion_signal_at_this_scale"

OUT_JSON.write_text(json.dumps({
    "ok": True,
    "decision": decision,
    "nonoverlap_to_overlap_rn_improvement": improvement,
    "cases": cases,
}, indent=2), encoding="utf-8")

lines = []
lines.append("# MULTIVERSION_RLBWT_PROBE_V1")
lines.append("")
lines.append("Status: measured")
lines.append("Date: 2026-06-28")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("Test the remaining RLBWT hypothesis: RLBWT may be weak for a single corpus but useful for a collection of similar versions.")
lines.append("")
lines.append("This is closer to pan-genome / many-similar-copy structure than to one isolated file.")
lines.append("")
lines.append("## Important boundary")
lines.append("")
lines.append("This test measures BWT run ratio only. It does not claim collection-safe retrieval semantics.")
lines.append("")
lines.append("A production multiversion GLYPH would need boundary-aware document/version semantics so matches cannot falsely cross version boundaries.")
lines.append("")
lines.append("## Setup")
lines.append("")
lines.append(f"- source: `{non['source']}`")
lines.append(f"- source sha256: `{non['source_sha256']}`")
lines.append(f"- versions: {non['versions']}")
lines.append(f"- version size: {non['version_mib']} MiB")
lines.append("")
lines.append("Two collections were built:")
lines.append("")
lines.append("- `nonoverlap`: adjacent chunks, weak version similarity baseline")
lines.append("- `overlap`: shifted overlapping chunks, synthetic rolling-version / backup-like probe")
lines.append("")
lines.append("## Results")
lines.append("")
lines.append("| case | collection bytes | shift MiB | BWT bytes | runs | r/n | avg run | classification |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
for c in cases:
    lines.append(
        f"| {c['label']} | {c['collection_bytes']} | {c['shift_mib']} | "
        f"{c['bwt_bytes']} | {c['runs']} | {c['r_over_n']:.8f} | "
        f"{c['avg_run_len']:.3f} | {c['classification']} |"
    )

lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- nonoverlap r/n: `{non['r_over_n']:.8f}`")
lines.append(f"- overlap r/n: `{ov['r_over_n']:.8f}`")
if improvement is not None:
    lines.append(f"- r/n improvement factor: `{improvement:.3f}x`")
lines.append("")
lines.append("## Decision")
lines.append("")
lines.append(f"`{decision}`")
lines.append("")
lines.append("Decision rule:")
lines.append("")
lines.append("- If overlap collection reaches `r/n <= 0.05` and improves strongly over nonoverlap, the multiversion RLBWT hypothesis is alive.")
lines.append("- If not, do not spend production engineering on RLBWT.")
lines.append("- Even with a positive signal, a real named multi-version corpus is required before public claims.")
lines.append("")
lines.append("## Non-claims")
lines.append("")
lines.append("- This does not prove production RLBWT memory usage.")
lines.append("- This does not prove collection-safe retrieval.")
lines.append("- This does not prove real backup/log/version workloads will behave the same.")
lines.append("- This is a gate test, not a product claim.")
lines.append("")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("wrote", OUT_MD)
print("wrote", OUT_JSON)
print("decision:", decision)
PY

echo
echo "=== report ==="
sed -n '1,220p' benchmarks/results/MULTIVERSION_RLBWT_PROBE_V1.md
