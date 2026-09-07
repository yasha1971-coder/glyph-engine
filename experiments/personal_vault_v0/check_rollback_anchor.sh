#!/usr/bin/env bash
set -euo pipefail
BASELINE="990d07b773197f747681754c171677934d4ee586"
BRANCH="personal-vault-v0-runtime-v2"

current="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
head="$(git rev-parse HEAD)"

echo "branch=$current"
echo "head=$head"
echo "baseline=$BASELINE"

git cat-file -e "$BASELINE^{commit}"

if git merge-base --is-ancestor "$BASELINE" HEAD; then
  echo "baseline_is_ancestor=true"
else
  echo "baseline_is_ancestor=false"
  exit 2
fi

echo "changed_paths_since_baseline:"
git diff --name-only "$BASELINE"..HEAD

echo "ROLLBACK_GUARD_OK"
