#!/usr/bin/env bash
set -euo pipefail
umask 022
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$(realpath -m "${1:?supply an output directory}")"
mkdir -p "$out"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
name=glyph-search-0.1.0-preview2
dest="$tmp/$name"
# Reuse the RC1 compiler targets, switches and license collection.
bash "$root/tools/glyph-evidence/build-release.sh" "$tmp/core"
core="$tmp/glyph-evidence-0.1.0-rc1"
tar -xzf "$tmp/core/glyph-evidence-0.1.0-rc1-linux-x86_64.tar.gz" -C "$tmp"
mkdir -p "$dest/tools"
cp -r "$core"/{build,licenses,LICENSE,THIRD_PARTY_NOTICES.md} "$dest/"
for file in build_glyph_index_v1.sh prepare_sentinel_corpus_v1.py export_rlbwt_full_runtime_v1.py rlbwt_container_v1.py rlbwt_rank_blocks_v1.py build_locate_fixture_v1.py recover_search_index_v1.py index_result_v1.py; do
    cp "$core/tools/$file" "$dest/tools/"
done
cp "$root/tools/glyph-search/"{glyph-search,README.md} "$dest/"
artifact="$name-linux-x86_64.tar.gz"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$tmp" -cf - "$name" | gzip -n > "$tmp/product.gz"
mv "$tmp/product.gz" "$out/$artifact"
(cd "$out" && sha256sum "$artifact" > "$artifact.sha256")
