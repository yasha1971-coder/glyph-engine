#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MINI_DIR="$ROOT/examples/mini"
OUT_DIR="$MINI_DIR/out"

mkdir -p "$OUT_DIR"

echo "[mini] corpus"
cp "$MINI_DIR/data.txt" "$OUT_DIR/corpus.bin"

echo "[mini] build SA32u"
./build/build_sa_u32 \
  "$OUT_DIR/corpus.bin" \
  "$OUT_DIR/corpus.sa.u32.bin"

echo "[mini] build BWT"
./build/build_bwt \
  "$OUT_DIR/corpus.bin" \
  "$OUT_DIR/corpus.sa.u32.bin" \
  "$OUT_DIR/corpus.bwt.bin" \
  0

echo "[mini] build FM"
./build/build_fm \
  "$OUT_DIR/corpus.bwt.bin" \
  "$OUT_DIR/corpus.fm.bin" \
  64

echo "[mini] query direct FM"
PATTERN_HEX="$(printf 'error' | xxd -p -c 999999)"

./build/query_fm_v1 \
  "$OUT_DIR/corpus.fm.bin" \
  "$OUT_DIR/corpus.bwt.bin" \
  "$PATTERN_HEX"

echo "[mini] done"
