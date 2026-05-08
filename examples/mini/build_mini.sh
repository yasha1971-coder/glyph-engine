#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MINI_DIR="$ROOT/examples/mini"
OUT_DIR="$MINI_DIR/out"

mkdir -p "$OUT_DIR"

echo "[mini] corpus"
cp "$MINI_DIR/data.txt" "$OUT_DIR/corpus.bin"

echo "[mini] build canonical sentinel-safe index"
"$ROOT/tools/build_glyph_index_v1.sh" \
  "$OUT_DIR/corpus.bin" \
  "$OUT_DIR"

echo "[mini] query direct FM"
PATTERN_HEX="$(printf 'error' | xxd -p -c 999999)"

./build/query_fm_v1 \
  "$OUT_DIR/fm.bin" \
  "$OUT_DIR/bwt.bin" \
  "$PATTERN_HEX"

echo "[mini] done"
