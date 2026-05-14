#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 tests/test_fm_cli_correctness.py
python3 tests/test_locate_verify.py
