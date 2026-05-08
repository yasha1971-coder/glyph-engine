#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: tools/build_glyph_index_v1.sh <raw_corpus> <out_dir>"
  exit 1
fi

RAW="$1"
OUT="$2"

mkdir -p "$OUT"

INDEX_CORPUS="$OUT/corpus.sentinel.bin"
SA="$OUT/sa.bin"
BWT="$OUT/bwt.bin"
FM="$OUT/fm.bin"

echo "[1/4] prepare sentinel corpus"
python3 tools/prepare_sentinel_corpus_v1.py \
  --input "$RAW" \
  --output "$INDEX_CORPUS"

echo "[2/4] build SA32u"
./build/build_sa_u32 "$INDEX_CORPUS" "$SA"

echo "[3/4] build BWT"
./build/build_bwt "$INDEX_CORPUS" "$SA" "$BWT" 0

echo "[4/4] build FM"
./build/build_fm "$BWT" "$FM" 128

echo "[done]"
echo "index_corpus: $INDEX_CORPUS"
echo "sa: $SA"
echo "bwt: $BWT"
echo "fm: $FM"
