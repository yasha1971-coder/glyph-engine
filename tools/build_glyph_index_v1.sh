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

echo "[5/5] write manifest"
python3 - <<PY
import hashlib, json
from pathlib import Path

raw = Path("$RAW")
index_corpus = Path("$INDEX_CORPUS")
sa = Path("$SA")
bwt = Path("$BWT")
fm = Path("$FM")
out = Path("$OUT")

def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "format": "GLYPH_INDEX_MANIFEST_V1",
    "raw_corpus": {
        "path": str(raw),
        "bytes": raw.stat().st_size,
        "sha256": sha256_file(raw),
    },
    "index_corpus": {
        "path": str(index_corpus),
        "bytes": index_corpus.stat().st_size,
        "sha256": sha256_file(index_corpus),
        "sentinel": "0x00",
    },
    "artifacts": {
        "sa": str(sa),
        "bwt": str(bwt),
        "fm": str(fm),
    },
}

(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("manifest:", out / "manifest.json")
PY
echo "[done]"
echo "index_corpus: $INDEX_CORPUS"
echo "sa: $SA"
echo "bwt: $BWT"
echo "fm: $FM"
