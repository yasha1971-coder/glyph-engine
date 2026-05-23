#!/usr/bin/env bash

set -e

CORPUS=${1:-enwik9}

echo "[1] SA"

time ./build/build_sa \
"$CORPUS" \
out/sa.bin

echo "[2] BWT"

time ./build/build_bwt \
"$CORPUS" \
out/sa.bin \
out/bwt.bin \
0x00

for STEP in 32 64 256
do

DIR="out_${STEP}"

mkdir -p "$DIR"

echo "[3] FM step=$STEP"

time ./build/build_fm \
out/bwt.bin \
"$DIR/fm.bin" \
"$STEP"

ls -lh "$DIR/fm.bin"

done

echo

echo "DONE"
