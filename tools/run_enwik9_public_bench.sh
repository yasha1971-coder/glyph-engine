#!/usr/bin/env bash

set -e

CORPUS=${1:-enwik9}
OUT_DIR=${2:-out_sentinel}

mkdir -p "$OUT_DIR"

echo "[0] sentinel-safe public bench"
echo "corpus:  $CORPUS"
echo "out_dir: $OUT_DIR"

echo "[1] SA sentinel-safe"

time ./build/build_sa_sentinel_v1 \
"$CORPUS" \
"$OUT_DIR/sa.bin"

echo "[2] BWT sentinel-safe"

time ./build/build_bwt_sentinel_v1 \
"$CORPUS" \
"$OUT_DIR/sa.bin" \
"$OUT_DIR/bwt.bin"

for STEP in 32 64 256
do

DIR="${OUT_DIR}_${STEP}"

mkdir -p "$DIR"

echo "[3] FM step=$STEP"

time ./build/build_fm \
"$OUT_DIR/bwt.bin" \
"$DIR/fm.bin" \
"$STEP"

ls -lh "$DIR/fm.bin"

done

echo

echo "DONE"