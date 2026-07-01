#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SILESIA_DIR="${1:-}"

mkdir -p benchmarks/results benchmarks/work/anchor_repeat_distance_profile_v1 docs/review
grep -qxF "benchmarks/work/" .gitignore || echo "benchmarks/work/" >> .gitignore

if [ -n "$SILESIA_DIR" ]; then
  python3 tools/anchor_repeat_distance_profile_v1.py "$SILESIA_DIR"
else
  python3 tools/anchor_repeat_distance_profile_v1.py
fi
