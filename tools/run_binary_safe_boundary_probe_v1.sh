#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/examples/binary-safe-boundary/out"

rm -rf "$WORK"
mkdir -p "$WORK"

CORPUS="$WORK/corpus_with_nul.bin"
INDEX="$WORK/index"
LOG="$WORK/boundary_probe.log"

python3 - <<'PY' "$CORPUS"
from pathlib import Path
import sys

p = Path(sys.argv[1])
data = bytes([0x41, 0x00, 0x42, 0x00, 0x43, 0x00, 0x42])
p.write_bytes(data)

print("corpus:", p)
print("corpus_bytes:", len(data))
print("corpus_hex:", data.hex())
print("query_hex:", bytes([0x00, 0x42]).hex())
print("expected_future_match_count:", 2)
print("expected_future_offsets:", [1, 5])
PY

set +e
(
  cd "$ROOT"
  bash tools/build_glyph_index_v1.sh "$CORPUS" "$INDEX"
) >"$LOG" 2>&1
RC=$?
set -e

echo "builder_exit_code: $RC"
echo "log: $LOG"

if [ "$RC" -eq 0 ]; then
  echo "BOUNDARY_PROBE_UNEXPECTED_PASS"
  echo "Current sentinel-safe builder accepted a corpus containing 0x00."
  echo "This requires immediate investigation before binary-safe claims."
  exit 1
fi

echo "BOUNDARY_PROBE_EXPECTED_FAIL"
echo "Current GLYPH v0.x rejects or fails on source corpus containing 0x00."
echo "This confirms the documented sentinel-safe boundary."
