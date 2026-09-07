#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-dist}"
ROOT="$(git rev-parse --show-toplevel)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
SHORT="${SHA:0:12}"
EPOCH="$(git -C "$ROOT" show -s --format=%ct "$SHA")"
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
NAME="glyph-v1-${OS}-${ARCH}-${SHORT}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
STAGE="$TMP/$NAME"
mkdir -p "$STAGE" "$OUTDIR"

cd "$ROOT"

# One product archive: V0.1 exact Vault lifecycle plus V1 authenticated retrieval.
# Corpora, sidecars, caches, model weights and materialized user data are excluded.
FILES=(
  CMakeLists.txt
  src
  third_party/libsais
  tools/rlbwt_query_v2.py
  tools/rlbwt_container_v2.py
  experiments/personal_vault_v0/glyph_vault_cli_v0.py
  experiments/personal_vault_v0/glyph_vault_cli_v1.py
  experiments/personal_vault_v0/real_intake_v1.py
  experiments/personal_vault_v0/vault_v0.py
  experiments/personal_vault_v0/rlb3x_fixed.py
  experiments/personal_vault_v0/loc2_experimental.py
  experiments/personal_vault_v0/query_loc2_experimental.py
  experiments/personal_vault_v0/query_rlb3x_count.py
  experiments/personal_vault_v0/query_rlb3x_loc2.py
  experiments/personal_vault_v0/query_rlb3x_object.py
  experiments/personal_vault_v0/restore_rlb3x.py
  deploy/personal-vault-v0.1/glyph.py
  deploy/glyph-v1
)

git archive "$SHA" "${FILES[@]}" | tar -x -C "$STAGE"

cmake -S "$STAGE" -B "$TMP/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$TMP/build" --target build_sa_binary_v1 build_bwt_binary_v1 -j2 >/dev/null
mkdir -p "$STAGE/deploy/personal-vault-v0.1/bin"
install -m 0755 "$TMP/build/build_sa_binary_v1" "$STAGE/deploy/personal-vault-v0.1/bin/"
install -m 0755 "$TMP/build/build_bwt_binary_v1" "$STAGE/deploy/personal-vault-v0.1/bin/"

CXX_ID="$(c++ --version | head -n1)"
CMAKE_ID="$(cmake --version | head -n1)"
python3 - "$STAGE" "$SHA" "$OS" "$ARCH" "$EPOCH" "$CXX_ID" "$CMAKE_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path

stage=Path(sys.argv[1])
release={
    'format':'GLYPH_RELEASE_INFO_V1',
    'version':'1',
    'git_sha':sys.argv[2],
    'platform':sys.argv[3],
    'arch':sys.argv[4],
    'source_commit_unix':int(sys.argv[5]),
    'compiler':sys.argv[6],
    'cmake':sys.argv[7],
    'source_deletion_enabled':False,
    'human_selection_required':True,
    'model_truth_authority':False,
    'production_claimed':False,
    'product_scope':[
        'init','add','verify','status','list','search_exact','restore',
        'free_space_dry_run','authenticated_query','human_select',
        'authenticated_materialize'
    ],
}
release_path=stage/'deploy/glyph-v1/RELEASE.json'
release_path.write_text(json.dumps(release,sort_keys=True,separators=(',',':'))+'\n')

rows=[]
for path in sorted(item for item in stage.rglob('*') if item.is_file()):
    relative=path.relative_to(stage).as_posix()
    if relative=='SHA256SUMS':
        continue
    rows.append((hashlib.sha256(path.read_bytes()).hexdigest(),relative))
with (stage/'SHA256SUMS').open('w') as output:
    for digest,relative in rows:
        output.write(f'{digest}  {relative}\n')
PY

( cd "$STAGE" && sha256sum -c SHA256SUMS >/dev/null )
python3 "$STAGE/deploy/glyph-v1/portable_runtime.py" verify-profile >/dev/null

TARBALL="$OUTDIR/$NAME.tar.gz"
tar --sort=name --mtime="@$EPOCH" --owner=0 --group=0 --numeric-owner \
  -C "$TMP" -cf - "$NAME" | gzip -n > "$TARBALL"
( cd "$OUTDIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
printf '%s\n' "$TARBALL"
