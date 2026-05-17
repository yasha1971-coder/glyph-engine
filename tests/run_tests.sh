#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 tests/test_fm_cli_correctness.py
python3 tests/test_locate_verify.py
python3 tests/test_manifest_integrity.py
python3 tests/test_query_verified.py
python3 tests/test_sa_container_v1.py
python3 tests/test_sa_container_reader_v1.py
