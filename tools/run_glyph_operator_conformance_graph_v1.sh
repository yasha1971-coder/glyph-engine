#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$ROOT"

RESULT="$ROOT/benchmarks/results/GLYPH_OPERATOR_CONFORMANCE_GRAPH_V1.json"

STDOUT_LOG="/tmp/GLYPH_OPERATOR_CONFORMANCE_GRAPH_V1.stdout.json"

echo "[operator-graph] GLYPH real-world operator path"

python3 -I \
  tools/check_operator_conformance_graph_v1.py \
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
    raise SystemExit(
        "operator conformance graph is not ok"
    )

if result.get("operator_conformant") is not True:
    raise SystemExit(
        "operator_conformant is not true"
    )

if result.get("verify_ok_permitted") is not True:
    raise SystemExit(
        "operator graph does not permit VERIFY OK"
    )

if result.get("node_count") != 6:
    raise SystemExit(
        "operator node count is not six"
    )

for node in result["nodes"]:
    print(
        f'{node["id"]} PASS — {node["name"]}'
    )

print("GLYPH OPERATOR CONFORMANCE OK")
PY
