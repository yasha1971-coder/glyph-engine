#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[verify] GLYPH one-command verification"

REQUIRED_BINS=(
  build/build_sa_u32
  build/build_bwt
  build/build_fm
  build/query_fm_v1
)

need_build=0

for bin in "${REQUIRED_BINS[@]}"; do
  if [[ ! -x "$bin" ]]; then
    need_build=1
  fi
done

if [[ "$need_build" -eq 1 ]]; then
  for cmd in cmake make c++ python3 xxd; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "VERIFY FAIL: missing required command: $cmd"
      echo "Install build tools, Python 3, and xxd; then run ./verify.sh again."
      exit 1
    fi
  done

  echo "[verify] building required binaries"

  cmake -S . -B build

  cmake --build build --target \
    build_sa_u32 \
    build_bwt \
    build_fm \
    query_fm_v1
else
  echo "[verify] required binaries found"
fi

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

"$ROOT/examples/mini/run_mini.sh" > "$OUT"

if grep -q "count:[[:space:]]*2" "$OUT"; then
  echo "[verify] building RLBWT locate server"
cmake --build build --target rlbwt_full_query_locate_server_v1

echo "[verify] RLBWT bounded evidence tiny fixture"
./tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh


echo "[verify] Structural Fingerprint V0 replay smoke"

SF_SOURCE="examples/rlbwt-bounded-evidence-tiny/out/corpus.bin"
SF_BWT="examples/rlbwt-bounded-evidence-tiny/out/index/bwt.bin"
SF_ARTIFACT="examples/rlbwt-bounded-evidence-tiny/out/structural_fingerprint_v0.json"
SF_REPLAY="examples/rlbwt-bounded-evidence-tiny/out/structural_fingerprint_replay_v0.json"

python3 tools/glyph_structural_fingerprint_v0.py "$SF_SOURCE" \
  --bwt-path "$SF_BWT" \
  --out "$SF_ARTIFACT" >/dev/null

python3 tools/replay_structural_fingerprint_v0.py "$SF_ARTIFACT" \
  --out "$SF_REPLAY" >/dev/null

python3 - <<'PY2'
import json
from pathlib import Path

p = Path("examples/rlbwt-bounded-evidence-tiny/out/structural_fingerprint_replay_v0.json")
j = json.loads(p.read_text())

if not j.get("ok"):
    raise SystemExit(f"structural fingerprint replay failed: {j.get('errors')}")

print("[verify] structural fingerprint replay ok")
PY2


echo "[verify] GLYPH proof graph P1-P12"
./tools/run_glyph_proof_graph_v1.sh

echo "[verify] GLYPH runtime conformance graph"
./tools/run_glyph_runtime_conformance_graph_v1.sh

echo "[verify] GLYPH operator conformance graph"
./tools/run_glyph_operator_conformance_graph_v1.sh
echo "[verify] GLYPH embedded I0 contract"
PYTHONDONTWRITEBYTECODE=1 python3 -I tools/check_embedded_i0_contract_v1.py --verify
echo "VERIFY OK"

echo ""
echo "If this worked for you, tell us your use case:"
echo "Issue: https://github.com/yasha1971-coder/glyph-engine/issues/3"
echo "Discussion: https://github.com/yasha1971-coder/glyph-engine/discussions/4"

  exit 0
fi

echo "VERIFY FAIL"
cat "$OUT"
exit 1