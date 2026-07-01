#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tools/audit_anchor_repeat_distance_predictor_v1.py
