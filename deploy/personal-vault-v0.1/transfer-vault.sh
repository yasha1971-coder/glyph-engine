#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  transfer-vault.sh <local-vault> <user@host> <remote-vault-path>

Example:
  transfer-vault.sh ~/GlyphVault glyph@ovh.example /srv/glyph/vaults/pilot-001

Requirements:
  - local `glyph`, ssh and rsync available
  - identical GLYPH v0.1 release installed on the remote host
  - remote destination path MUST NOT already exist

This is a fail-closed first-replica transfer. It never deletes the local Vault.
EOF
  exit 2
}

[[ $# -eq 3 ]] || usage
LOCAL_VAULT="$1"
REMOTE="$2"
REMOTE_VAULT="$3"
GLYPH_LOCAL="${GLYPH_LOCAL:-glyph}"
GLYPH_REMOTE="${GLYPH_REMOTE:-$HOME/.local/bin/glyph}"

for x in "$GLYPH_LOCAL" ssh rsync python3; do
  command -v "$x" >/dev/null 2>&1 || { echo "missing local command: $x" >&2; exit 2; }
done
[[ -d "$LOCAL_VAULT" ]] || { echo "local Vault not found: $LOCAL_VAULT" >&2; exit 2; }

# Full local verification is mandatory before any replica is created.
"$GLYPH_LOCAL" verify "$LOCAL_VAULT" >/tmp/glyph-transfer-local-verify.json
LOCAL_STATUS_BEFORE="$($GLYPH_LOCAL status "$LOCAL_VAULT")"
LOCAL_ROOT="$(python3 -c 'import json,sys; x=json.loads(sys.stdin.read()); print(x.get("latest_root_sha256") or "")' <<<"$LOCAL_STATUS_BEFORE")"
[[ ${#LOCAL_ROOT} -eq 64 ]] || { echo "local Vault has no committed root" >&2; exit 2; }
TRANSFER_ID="${LOCAL_ROOT:0:16}"
REMOTE_STAGE="${REMOTE_VAULT}.incoming-${TRANSFER_ID}"

# Confirm the exact same product release is active on both machines.
LOCAL_VERSION="$($GLYPH_LOCAL version)"
REMOTE_VERSION="$(ssh "$REMOTE" "$GLYPH_REMOTE version")"
LOCAL_RELEASE_SHA="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("git_sha") or "")' <<<"$LOCAL_VERSION")"
REMOTE_RELEASE_SHA="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("git_sha") or "")' <<<"$REMOTE_VERSION")"
[[ -n "$LOCAL_RELEASE_SHA" && "$LOCAL_RELEASE_SHA" == "$REMOTE_RELEASE_SHA" ]] || {
  echo "release mismatch: local=$LOCAL_RELEASE_SHA remote=$REMOTE_RELEASE_SHA" >&2
  exit 3
}

ssh "$REMOTE" "set -eu; test ! -e '$REMOTE_VAULT'; rm -rf '$REMOTE_STAGE'; mkdir -p '$REMOTE_STAGE'"

# Canonical repository state is copied. Disposable cache/derived/journal content is rebuilt remotely.
rsync -a --partial --human-readable \
  --exclude='/cache/***' \
  --exclude='/journal/***' \
  --exclude='/derived/***' \
  --exclude='/quarantine/***' \
  "$LOCAL_VAULT/" "$REMOTE:$REMOTE_STAGE/"

ssh "$REMOTE" "set -eu; mkdir -p '$REMOTE_STAGE/cache' '$REMOTE_STAGE/journal' '$REMOTE_STAGE/derived/text' '$REMOTE_STAGE/derived/metadata' '$REMOTE_STAGE/derived/ai' '$REMOTE_STAGE/quarantine'"

# Reject a moving source Vault. V0.1 intentionally has no online snapshot-transfer protocol yet.
LOCAL_STATUS_AFTER="$($GLYPH_LOCAL status "$LOCAL_VAULT")"
LOCAL_ROOT_AFTER="$(python3 -c 'import json,sys; x=json.loads(sys.stdin.read()); print(x.get("latest_root_sha256") or "")' <<<"$LOCAL_STATUS_AFTER")"
[[ "$LOCAL_ROOT_AFTER" == "$LOCAL_ROOT" ]] || {
  echo "local Vault changed during transfer; remote staging left unpublished: $REMOTE_STAGE" >&2
  exit 4
}

# Remote full restore/hash verification is the publication gate.
REMOTE_VERIFY="$(ssh "$REMOTE" "$GLYPH_REMOTE verify '$REMOTE_STAGE'")"
REMOTE_STATUS="$(ssh "$REMOTE" "$GLYPH_REMOTE status '$REMOTE_STAGE'")"
REMOTE_ROOT="$(python3 -c 'import json,sys; x=json.loads(sys.stdin.read()); print(x.get("latest_root_sha256") or "")' <<<"$REMOTE_STATUS")"
[[ "$REMOTE_ROOT" == "$LOCAL_ROOT" ]] || {
  echo "remote root mismatch after verification: local=$LOCAL_ROOT remote=$REMOTE_ROOT" >&2
  exit 5
}

# Atomic publication: an unverified staging directory is never exposed as the destination Vault.
ssh "$REMOTE" "set -eu; test ! -e '$REMOTE_VAULT'; mv '$REMOTE_STAGE' '$REMOTE_VAULT'"

cat <<EOF
GLYPH_VAULT_REPLICA_OK
release_git_sha=$LOCAL_RELEASE_SHA
root_sha256=$LOCAL_ROOT
local_vault=$LOCAL_VAULT
remote=$REMOTE
remote_vault=$REMOTE_VAULT
source_deleted=false
remote_full_verify=true
EOF
