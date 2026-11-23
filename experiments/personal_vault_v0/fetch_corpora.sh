#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="https://github.com/zlib-ng/corpora.git"
SOURCE_COMMIT="5583ca94d1643b6dcd6b6dd2ad0c5704a4afa094"
OUT="${1:-.personal-vault-corpora-v0}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --quiet --filter=blob:none --no-checkout "$SOURCE_REPO" "$TMP/corpora"
git -C "$TMP/corpora" checkout --quiet "$SOURCE_COMMIT" --   canterbury calgary artificial miscellaneous

rm -rf "$OUT"
mkdir -p "$OUT"
for corpus in canterbury calgary artificial miscellaneous; do
  cp -a "$TMP/corpora/$corpus" "$OUT/$corpus"
done

count="$(find "$OUT" -type f | wc -l | tr -d ' ')"
bytes="$(find "$OUT" -type f -print0 | xargs -0 stat -c '%s' | awk '{s+=$1} END{print s+0}')"

test "$count" = "30"
test "$bytes" = "7252407"

(
  cd "$OUT"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$OUT/SHA256SUMS"

echo "source_repo=$SOURCE_REPO"
echo "source_commit=$SOURCE_COMMIT"
echo "file_count=$count"
echo "payload_bytes=$bytes"
echo "sha256_manifest=$OUT/SHA256SUMS"
