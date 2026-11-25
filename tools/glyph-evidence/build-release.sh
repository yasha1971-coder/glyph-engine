#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$(realpath -m "${1:?supply an output directory}")"
mkdir -p "$out"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
name=glyph-evidence-0.1.0-rc1
dest="$tmp/$name"
mkdir -p "$dest"/{tools,build,licenses}
cd "$root"
test "$(uname -sm)" = 'Linux x86_64'
cp LICENSE THIRD_PARTY_NOTICES.md tools/glyph-evidence/{glyph-evidence,README.md} "$dest/"
cp third_party/libsais/libsais/LICENSE "$dest/licenses/libsais-LICENSE"
for tool in build_glyph_index_v1.sh prepare_sentinel_corpus_v1.py export_rlbwt_full_runtime_v1.py rlbwt_container_v1.py rlbwt_rank_blocks_v1.py build_locate_fixture_v1.py make_rlbwt_bounded_evidence_v1.py make_rlbwt_bounded_evidence_bundle_v1.py verify_rlbwt_bounded_evidence_bundle_v1.py verify_rlbwt_bounded_evidence_v1.py; do
    cp "tools/$tool" "$dest/tools/"
done
inc=third_party/libsais/libsais/include
for unit in libsais libsais64; do
    gcc -std=c99 -O2 -I"$inc" -c "third_party/libsais/libsais/src/$unit.c" -o "$tmp/$unit.o"
done
g++ -std=c++17 -O2 -static -s -I"$inc" src/build_sa_u32.cpp "$tmp/libsais.o" "$tmp/libsais64.o" -o "$dest/build/build_sa_u32"
for unit in build_bwt build_fm rlbwt_full_query_locate_server_v1; do
    g++ -std=c++17 -O2 -static -s "src/$unit.cpp" -o "$dest/build/$unit"
done
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$tmp" -cf - "$name" | gzip -n > "$tmp/release.tar.gz"
artifact="$name-linux-x86_64.tar.gz"
mv "$tmp/release.tar.gz" "$out/$artifact"
(cd "$out" && sha256sum "$artifact" > "$artifact.sha256")
