#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 tests/test_fm_cli_correctness.py
python3 tests/test_locate_verify.py
python3 tests/test_manifest_integrity.py
python3 tests/test_query_verified.py
python3 tests/test_sa_container_v1.py
python3 tests/test_sa_container_reader_v1.py
python3 tests/test_golden_queries_v1.py
python3 tests/test_query_protocol_v1.py
python3 tests/test_query_server_protocol_v1.py
python3 tests/test_http_query_protocol_v1.py
python3 tests/test_capability_probe_v1.py

python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v
