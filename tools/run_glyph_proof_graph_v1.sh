#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RESULT="benchmarks/results/GLYPH_PROOF_GRAPH_V1.json"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

echo "[proof-graph] GLYPH P1-P12"

python3 -I tools/check_verify_chain_v1.py \
  --result "$RESULT" >"$TMP"

python3 - "$TMP" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())

if result.get("ok") is not True:
    raise SystemExit("GLYPH proof graph failed")

if result.get("proof_count") != 12:
    raise SystemExit("proof count is not 12")

if result.get("passed") != 12:
    raise SystemExit("passed count is not 12")

if result.get("failed") != 0:
    raise SystemExit("failed count is not zero")

if result.get("verify_ok_permitted") is not True:
    raise SystemExit("VERIFY OK is not permitted")

proofs = result.get("proofs")

if not isinstance(proofs, list):
    raise SystemExit("proof list missing")

expected = [f"P{i}" for i in range(1, 13)]
actual = [proof.get("id") for proof in proofs]

if actual != expected:
    raise SystemExit(
        f"proof sequence mismatch: expected={expected}, actual={actual}"
    )

for proof in proofs:
    if proof.get("status") != "PASS":
        raise SystemExit(f"{proof.get('id')} did not pass")
    print(f"{proof['id']} PASS — {proof['name']}")

print("GLYPH PROOF GRAPH OK")
PY
