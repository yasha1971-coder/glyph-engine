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
  echo "[verify] RLBWT bounded evidence tiny fixture"
./tools/run_rlbwt_bounded_evidence_tiny_fixture_v1.sh

echo "VERIFY OK"

echo ""
echo "If this worked for you, tell us your use case:"
echo "https://github.com/yasha1971-coder/glyph-engine/issues/new/choose"

  exit 0
fi

echo "VERIFY FAIL"
cat "$OUT"
exit 1