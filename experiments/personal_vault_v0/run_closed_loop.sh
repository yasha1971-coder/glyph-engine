#!/usr/bin/env bash
set -Eeuo pipefail
export PS4='+ [PV0 ${LINENO}] '
trap 'rc=$?; echo "PV0_FAIL rc=$rc line=$LINENO command=$BASH_COMMAND" >&2; exit $rc' ERR
set -x
ROOT="${1:-/tmp/pv0}"
rm -rf "$ROOT"; mkdir -p "$ROOT"
bash experiments/personal_vault_v0/fetch_corpora.sh "$ROOT/input"
python3 experiments/personal_vault_v0/vault_v0.py pack "$ROOT/input" "$ROOT/corpus.bin" "$ROOT/objects.json"
cmake -S . -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" --target build_sa_binary_v1 build_bwt_binary_v1 -j2 >/dev/null
"$ROOT/build/build_sa_binary_v1" "$ROOT/corpus.bin" "$ROOT/sa.bin"
"$ROOT/build/build_bwt_binary_v1" "$ROOT/corpus.bin" "$ROOT/sa.bin" "$ROOT/bwt.bin"
python3 tools/rlbwt_container_v2.py encode "$ROOT/bwt.bin" "$ROOT/bwt.rlb2"
python3 tools/rlbwt_rank_v2.py build "$ROOT/bwt.rlb2" "$ROOT/bwt.rlr2" --rank-step 8192
N="$(stat -c %s "$ROOT/corpus.bin")"; ROWS=$((N+1))
python3 experiments/personal_vault_v0/vault_v0.py locate "$ROOT/sa.bin" "$ROWS" 128 "$ROOT/locate.bin"
python3 experiments/personal_vault_v0/vault_v0.py restore-bwt "$ROOT/bwt.bin" "$ROOT/restored.bin"
cmp "$ROOT/corpus.bin" "$ROOT/restored.bin"
python3 experiments/personal_vault_v0/space_frontier_probe.py "$ROOT/corpus.bin" "$ROOT/bwt.bin" "$ROOT/bwt.rlb2" "$ROOT/bwt.rlr2" "$ROOT/locate.bin" "$ROOT/space-frontier.json"
python3 experiments/personal_vault_v0/aux_frontier_probe.py "$ROOT/corpus.bin" "$ROOT/bwt.bin" "$ROOT/aux-frontier.json"
python3 experiments/personal_vault_v0/vault_v0.py verify-objects "$ROOT/restored.bin" "$ROOT/objects.json" "$ROOT/input"
python3 experiments/personal_vault_v0/vault_v0.py queries "$ROOT/corpus.bin" "$ROOT/objects.json" "$ROOT/queries.json"
python3 experiments/personal_vault_v0/vault_v0.py boundaries "$ROOT/corpus.bin" "$ROOT/objects.json" "$ROOT/boundaries.json"
python3 - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); c=(r/"corpus.bin").read_bytes(); m=json.loads((r/"objects.json").read_text())
files={"rlb2":r/"bwt.rlb2","rlr2":r/"bwt.rlr2","locate":r/"locate.bin"}
manifest={"format":"GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2","version":1,"corpus_identity":{"reference_id":"personal-vault-v0-30-object-concat","bytes":len(c),"md5":hashlib.md5(c).hexdigest(),"sha256":hashlib.sha256(c).hexdigest()},"row_count":len(c)+1,"rank_step":8192,"sample_step":128,"runtime_data_bytes":sum(p.stat().st_size for p in files.values()),"files":{}}
formats={"rlb2":"GLYPH_RLB2_EXPERIMENTAL_V2","rlr2":"GLYPH_RLR2_V2","locate":"LOC1"}
for k,p in files.items(): manifest["files"][k]={"name":p.name,"format":formats[k],"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(r/"runtime.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
PY
python3 - "$ROOT" <<'PY'
import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]); qs=json.loads((r/"queries.json").read_text())["queries"]
assert len(qs)>=20, len(qs)
for i,q in enumerate(qs):
    out=subprocess.check_output(["python3","tools/rlbwt_query_v2.py","--runtime-manifest",str(r/"runtime.json"),"--pattern-hex",q["pattern_hex"],"--max-offsets","-1"],text=True)
    result=json.loads(out.splitlines()[0])
    assert result["count"]==1 and result["locate_offsets"]==[q["expected_offset"]], (q,result)
bounds=json.loads((r/"boundaries.json").read_text())["cases"]
assert len(bounds)>=20, len(bounds)
rejected=0
for q in bounds:
    out=subprocess.check_output(["python3","tools/rlbwt_query_v2.py","--runtime-manifest",str(r/"runtime.json"),"--pattern-hex",q["pattern_hex"],"--max-offsets","-1"],text=True)
    result=json.loads(out.splitlines()[0])
    # Raw concatenation MUST see the cross-object byte string. Vault semantics must reject it.
    if q["forbidden_offset"] in result["locate_offsets"]: rejected+=1
assert rejected==len(bounds)
runtime=json.loads((r/"runtime.json").read_text())
report={"format":"GLYPH_PERSONAL_VAULT_V0_CLOSED_LOOP","objects":len(json.loads((r/"objects.json").read_text())["objects"]),"source_bytes":runtime["corpus_identity"]["bytes"],"runtime_data_bytes":runtime["runtime_data_bytes"],"runtime_ratio":runtime["runtime_data_bytes"]/runtime["corpus_identity"]["bytes"],"restore_sha256_equal":True,"object_restore_equal":True,"exact_unique_queries":len(qs),"exact_unique_queries_passed":len(qs),"cross_object_boundary_candidates":len(bounds),"cross_object_boundary_candidates_seen_by_raw_glyph":rejected,"vault_boundary_filter_required":True}
(r/"report.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
print(json.dumps(report,sort_keys=True))
PY
