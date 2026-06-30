#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SILESIA_DIR="${1:-}"

if [ -n "$SILESIA_DIR" ]; then
  python3 tools/silesia_per_file_rlbwt_profile_v1.py "$SILESIA_DIR"
else
  python3 tools/silesia_per_file_rlbwt_profile_v1.py
fi
