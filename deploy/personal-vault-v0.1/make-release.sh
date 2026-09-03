#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-dist}"
ROOT="$(git rev-parse --show-toplevel)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
SHORT="${SHA:0:12}"
EPOCH="$(git -C "$ROOT" show -s --format=%ct "$SHA")"
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
NAME="glyph-personal-vault-v0.1-${OS}-${ARCH}-${SHORT}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
STAGE="$TMP/$NAME"
mkdir -p "$STAGE" "$OUTDIR"

cd "$ROOT"

# Product-only source whitelist. Research models and rejected semantic experiments
# are intentionally excluded from the distributable.
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
  deploy/personal-vault-v0.1/install.sh
  deploy/personal-vault-v0.1/transfer-vault.sh
  deploy/personal-vault-v0.1/README.md
)

git archive "$SHA" "${FILES[@]}" | tar -x -C "$STAGE"

# Build only the two construction helpers used by the V0.1 product path.
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
stage=Path(sys.argv[1]); sha=sys.argv[2]; os_name=sys.argv[3]; arch=sys.argv[4]
epoch=int(sys.argv[5]); cxx=sys.argv[6]; cmake=sys.argv[7]
rows=[]
for p in sorted(x for x in stage.rglob('*') if x.is_file()):
    rel=p.relative_to(stage).as_posix()
    if rel.endswith('RELEASE.json') or rel.endswith('SHA256SUMS'): continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append({'path':rel,'bytes':p.stat().st_size,'sha256':h})
release={
  'format':'GLYPH_RELEASE_INFO_V0_1',
  'version':'0.1',
  'git_sha':sha,
  'platform':os_name,
  'arch':arch,
  'source_commit_unix':epoch,
  'compiler':cxx,
  'cmake':cmake,
  'source_deletion_enabled':False,
  'product_scope':['init','add','verify','status','list','search_exact','restore','free-space-dry-run'],
  'file_count':len(rows),
}
(stage/'deploy/personal-vault-v0.1/RELEASE.json').write_text(json.dumps(release,sort_keys=True,separators=(',',':'))+'\n')
with (stage/'SHA256SUMS').open('w') as f:
    for r in rows:
        f.write(f"{r['sha256']}  {r['path']}\n")
p=stage/'deploy/personal-vault-v0.1/RELEASE.json'
with (stage/'SHA256SUMS').open('a') as f:
    f.write(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  deploy/personal-vault-v0.1/RELEASE.json\n')
PY

# Self-check manifest before packaging.
( cd "$STAGE" && sha256sum -c SHA256SUMS >/dev/null )

# Normalize archive metadata. Build provenance is additionally bound by the
# manual GitHub attestation workflow.
TARBALL="$OUTDIR/$NAME.tar.gz"
tar --sort=name --mtime="@$EPOCH" --owner=0 --group=0 --numeric-owner -C "$TMP" -cf - "$NAME" | gzip -n > "$TARBALL"
sha256sum "$TARBALL" > "$TARBALL.sha256"
printf '%s\n' "$TARBALL"
