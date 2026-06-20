#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SRC_CORPUS="REFERENCE_BENCH/OUT/pizza_sentinel_test/english_50mb.txt"
DEMO_DIR="examples/public-evidence-demo/work/pizza_english_50mb"
CORPUS="$DEMO_DIR/corpus.bin"
QUERY="Ten Days that Shook the World"
AUDIT="$DEMO_DIR/audit_ten_days_v0.json"
EVIDENCE="$DEMO_DIR/evidence_ten_days_v1.json"

echo "[public-demo] GLYPH Pizza & Chili English 50MB evidence demo"

if [ ! -f "$SRC_CORPUS" ]; then
  echo "[public-demo] missing source corpus:"
  echo "  $SRC_CORPUS"
  echo
  echo "This script expects the local Pizza & Chili English 50MB corpus prefix."
  echo "No large corpus is committed to git."
  exit 1
fi

mkdir -p "$DEMO_DIR"

if [ ! -f "$CORPUS" ]; then
  echo "[public-demo] copy corpus"
  cp "$SRC_CORPUS" "$CORPUS"
else
  echo "[public-demo] corpus already exists"
fi

echo "[public-demo] corpus properties"
wc -c "$CORPUS"

python3 - <<'PY'
from pathlib import Path
p = Path("examples/public-evidence-demo/work/pizza_english_50mb/corpus.bin")
d = p.read_bytes()
print("bytes:", len(d))
print("null_bytes:", d.count(b"\x00"))
if len(d) != 50_000_000:
    raise SystemExit("unexpected corpus size")
if d.count(b"\x00") != 0:
    raise SystemExit("corpus contains embedded 0x00; not valid for current v0.x sentinel model")
PY

echo "[public-demo] build canonical sentinel-safe index"
tools/build_glyph_index_v1.sh "$CORPUS" "$DEMO_DIR"

echo "[public-demo] build locate layer"
python3 tools/build_locate_fixture_v1.py \
  --bwt "$DEMO_DIR/bwt.bin" \
  --sa "$DEMO_DIR/sa.bin" \
  --out-dir "$DEMO_DIR" \
  --checkpoint-step 128 \
  --sample-step 16

echo "[public-demo] create Audit Artifact V0"
python3 tools/glyph_make_audit_artifact_v0.py \
  --index-dir "$DEMO_DIR" \
  --query "$QUERY" \
  --output "$AUDIT"

echo "[public-demo] verify Audit Artifact V0"
python3 tools/glyph_verify_audit_artifact_v0.py "$AUDIT"

echo "[public-demo] create Evidence Case V1"
python3 tools/glyph_make_evidence_case_v1.py \
  --artifact "$AUDIT" \
  --output "$EVIDENCE"

echo "[public-demo] check expected evidence result"
python3 - <<'PY'
import json
from pathlib import Path

audit = json.loads(Path("examples/public-evidence-demo/work/pizza_english_50mb/audit_ten_days_v0.json").read_text())
case = json.loads(Path("examples/public-evidence-demo/work/pizza_english_50mb/evidence_ten_days_v1.json").read_text())

result = audit["result"]
records = case["evidence_records"]

print("match_count:", result["match_count"])
print("fm_interval:", result["fm_interval"])
print("offset_mode:", result["offset_mode"])
print("offsets:", result["offsets"])
print("records:", len(records))

if result["match_count"] != 1:
    raise SystemExit("expected match_count=1")
if result["fm_interval"] != [12587658, 12587659]:
    raise SystemExit("unexpected FM interval")
if result["offset_mode"] != "locate_backend_v2":
    raise SystemExit("expected locate_backend_v2")
if result["offsets"] != [53]:
    raise SystemExit("expected offsets=[53]")
if len(records) != 1:
    raise SystemExit("expected one evidence record")
if records[0].get("byte_check") is not True:
    raise SystemExit("expected byte_check=true")
if "Ten Days that Shook the World" not in records[0].get("snippet_text", ""):
    raise SystemExit("expected query text in snippet")

print("byte_check:", records[0]["byte_check"])
print("snippet_contains_query: true")
PY

echo "[public-demo] PASS"
echo "[public-demo] audit:    $AUDIT"
echo "[public-demo] evidence: $EVIDENCE"
