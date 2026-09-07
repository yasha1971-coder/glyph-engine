#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$HERE/../.." && pwd)"
PREFIX="${GLYPH_PREFIX:-$HOME/.local/share/glyph}"
BINDIR="${GLYPH_BIN_DIR:-$HOME/.local/bin}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "GLYPH V1 currently supports Linux and WSL/Linux." >&2
  exit 2
fi

if [[ ! -f "$PACKAGE_ROOT/SHA256SUMS" || ! -f "$HERE/RELEASE.json" ]]; then
  echo "Not a packaged GLYPH V1 release: SHA256SUMS/RELEASE.json missing." >&2
  exit 2
fi

( cd "$PACKAGE_ROOT" && sha256sum -c SHA256SUMS >/dev/null )

SHA="$(python3 - "$HERE/RELEASE.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1], encoding='utf-8'))
sha=value.get('git_sha')
if not isinstance(sha,str) or len(sha)!=40:
    raise SystemExit('invalid release git_sha')
print(sha)
PY
)"

RELEASES="$PREFIX/releases"
DEST="$RELEASES/$SHA"
TMP="$RELEASES/.install-v1-$SHA-$$"
mkdir -p "$RELEASES" "$BINDIR"

if [[ -e "$DEST" ]]; then
  if [[ ! -f "$DEST/deploy/glyph-v1/portable_runtime.py" ]]; then
    echo "Existing release directory is not GLYPH V1: $DEST" >&2
    exit 3
  fi
  python3 "$DEST/deploy/glyph-v1/portable_runtime.py" verify-profile >/dev/null
  echo "Verified GLYPH V1 release already installed: $DEST"
else
  trap 'rm -rf "$TMP"' EXIT
  mkdir -p "$TMP"
  cp -a "$PACKAGE_ROOT/." "$TMP/"
  chmod 0755 "$TMP/deploy/glyph-v1/glyph.py" \
    "$TMP/deploy/glyph-v1/portable_runtime.py"
  chmod 0755 "$TMP/deploy/personal-vault-v0.1/glyph.py"
  chmod 0755 "$TMP/deploy/personal-vault-v0.1/bin/"* 2>/dev/null || true
  ( cd "$TMP" && sha256sum -c SHA256SUMS >/dev/null )
  python3 "$TMP/deploy/glyph-v1/portable_runtime.py" verify-profile >/dev/null
  mv "$TMP" "$DEST"
  trap - EXIT
fi

ln -sfn "$DEST/deploy/glyph-v1/glyph.py" "$BINDIR/glyph"

printf 'GLYPH V1 activated\n'
printf 'release: %s\n' "$SHA"
printf 'path:    %s\n' "$DEST"
printf 'command: %s/glyph\n' "$BINDIR"
"$BINDIR/glyph" version
"$BINDIR/glyph" verify-profile
