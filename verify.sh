#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[verify] GLYPH one-command verification"

need_build=0

for bin in \
  build/query_fm_v1 \
  build/build_sa_sentinel_v1 \
  build/build_bwt_sentinel_v1 \
  build/build_fm
do
  if [[ ! -x "$bin" ]]; then
    need_build=1
  fi
done

if [[ "$need_build" -eq 1 ]]; then

  for cmd in cmake make c++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "VERIFY FAIL: missing required command: $cmd"
      echo "Install build tools, then run ./verify.sh again."
      exit 1
    fi
  done

  echo "[verify] building required binaries"

  cmake -S . -B build

  cmake --build build --target \
    query_fm_v1 \
    build_sa_sentinel_v1 \
    build_bwt_sentinel_v1 \
    build_fm

else
  echo "[verify] required binaries found"
fi

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

"$ROOT/examples/mini/run_mini.sh" > "$OUT"

if grep -q "count:[[:space:]]*2" "$OUT"; then
  echo "VERIFY OK"
  exit 0
fi

echo "VERIFY FAIL"
cat "$OUT"
exit 1