#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SILESIA_DIR="${1:-/tmp/silesia_check}"
python3 tools/match_distance_profile_v1.py "$SILESIA_DIR"
