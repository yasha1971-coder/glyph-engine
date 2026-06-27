#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/examples/rlbwt-bounded-evidence-tiny/out"

rm -rf "$WORK"
mkdir -p "$WORK"

CORPUS="$WORK/corpus.bin"

cat > "$CORPUS" <<'EOF'
alpha beta gamma
the quick brown fox
delta the epsilon
the final line
EOF

# v0.x sentinel invariant: no NUL bytes in source corpus
python3 - <<'PY' "$CORPUS"
from pathlib import Path
import sys
p = Path(sys.argv[1])
data = p.read_bytes()
if b"\x00" in data:
    raise SystemExit("NUL byte found")
print("corpus_bytes:", len(data))
PY

INDEX="$WORK/index"
RUNTIME="$WORK/rlbwt_runtime"
ART="$WORK/rlbwt_bounded_evidence_v1.json"

# Build normal GLYPH index.
python3 "$ROOT/tools/build_glyph_index_v1.py" \
  --corpus "$CORPUS" \
  --out-dir "$INDEX"

# Export RLBWT full runtime.
python3 "$ROOT/tools/export_rlbwt_full_runtime_v1.py" \
  --source-index-dir "$INDEX" \
  --out-dir "$RUNTIME" \
  --rank-step 128 \
  --sample-step 16 \
  --force

# Create bounded evidence artifact.
python3 "$ROOT/tools/make_rlbwt_bounded_evidence_v1.py" \
  --runtime-dir "$RUNTIME" \
  --source-corpus "$CORPUS" \
  --query "the" \
  --max-offsets 2 \
  --out "$ART"

# Replay verify artifact.
python3 "$ROOT/tools/verify_rlbwt_bounded_evidence_v1.py" \
  --artifact "$ART"

python3 - <<'PY' "$ART"
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text())

assert j["artifact_version"] == "RLBWT_BOUNDED_EVIDENCE_V1"
assert j["profile"] == "RLBWT_FULL_RUNTIME_PROFILE_V1"
assert j["query"]["text"] == "the"
assert j["retrieval"]["match_count"] == 3
assert j["retrieval"]["max_offsets"] == 2
assert j["retrieval"]["returned_count"] == 2
assert j["retrieval"]["bounded"] is True
assert j["byte_check"]["all_returned_offsets_match_query"] is True
assert j["ok"] is True

print("[tiny-fixture] PASS")
print("artifact:", p)
print("fm_interval:", j["retrieval"]["fm_interval"])
print("match_count:", j["retrieval"]["match_count"])
print("offsets:", j["retrieval"]["offsets"])
PY
