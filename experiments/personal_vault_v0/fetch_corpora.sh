#!/usr/bin/env bash
set -euo pipefail
SOURCE_REPO="https://github.com/zlib-ng/corpora.git"
SOURCE_COMMIT="5583ca94d1643b6dcd6b6dd2ad0c5704a4afa094"
OUT="${1:-.personal-vault-corpora-v0}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "PV0_FETCH: clone pinned corpus mirror"
git clone --quiet --filter=blob:none --no-checkout "$SOURCE_REPO" "$TMP/corpora"

echo "PV0_FETCH: checkout exact classic file set"
git -C "$TMP/corpora" checkout --quiet "$SOURCE_COMMIT" --   canterbury artificial miscellaneous   calgary/bib calgary/book1 calgary/book2 calgary/geo calgary/news   calgary/obj1 calgary/obj2 calgary/paper1 calgary/paper2 calgary/pic   calgary/progc calgary/progl calgary/progp calgary/trans

rm -rf "$OUT"
mkdir -p "$OUT/calgary"
cp -a "$TMP/corpora/canterbury" "$OUT/canterbury"
cp -a "$TMP/corpora/artificial" "$OUT/artificial"
cp -a "$TMP/corpora/miscellaneous" "$OUT/miscellaneous"
for f in bib book1 book2 geo news obj1 obj2 paper1 paper2 pic progc progl progp trans; do
  cp -a "$TMP/corpora/calgary/$f" "$OUT/calgary/$f"
done

count="$(find "$OUT" -type f | wc -l | tr -d ' ')"
bytes="$(find "$OUT" -type f -print0 | xargs -0 stat -c '%s' | awk '{s+=$1} END{print s+0}')"

echo "PV0_FETCH: file_count=$count payload_bytes=$bytes"
test "$count" = "30"
test "$bytes" = "7230581"

(
  cd "$OUT"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$OUT/SHA256SUMS"

printf 'source_commit=%s\nfile_count=%s\npayload_bytes=%s\n' "$SOURCE_COMMIT" "$count" "$bytes"
