#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$ROOT"

RESULT="$ROOT/benchmarks/results/GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1.json"
STDOUT_LOG="/tmp/GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1.stdout.json"

echo "[runtime-graph] GLYPH binary-safe C++ runtime"

python3 -I tools/check_runtime_conformance_graph_v1.py \
  --result "$RESULT" \
  > "$STDOUT_LOG"

python3 - "$RESULT" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(
    Path(sys.argv[1]).read_text()
)

if result.get("ok") is not True:
    raise SystemExit("runtime conformance graph is not ok")

if result.get("runtime_conformant") is not True:
    raise SystemExit("runtime_conformant is not true")

if result.get("verify_ok_permitted") is not True:
    raise SystemExit("runtime graph does not permit VERIFY OK")

for node in result["nodes"]:
    print(
        f'{node["id"]} PASS — {node["name"]}'
    )

print("GLYPH RUNTIME CONFORMANCE OK")
PY
