#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[verify] GLYPH mini deterministic verification"

if [[ ! -x "$ROOT/examples/mini/run_mini.sh" ]]; then
  echo "VERIFY FAIL: examples/mini/run_mini.sh is not executable"
  exit 1
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
