#!/usr/bin/env bash
set -Eeuo pipefail
export PS4='+ [PV0-FAST ${LINENO}] '
trap 'rc=$?; echo "PV0_FAST_FAIL rc=$rc line=$LINENO command=$BASH_COMMAND" >&2; exit $rc' ERR
set -x
ROOT="${1:-/tmp/pv0}"
rm -rf "$ROOT"; mkdir -p "$ROOT"

bash experiments/personal_vault_v0/fetch_corpora.sh "$ROOT/input"
python3 experiments/personal_vault_v0/vault_v0.py pack "$ROOT/input" "$ROOT/corpus.bin" "$ROOT/objects.json"

cmake -S . -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" --target build_sa_binary_v1 build_bwt_binary_v1 -j2 >/dev/null
"$ROOT/build/build_sa_binary_v1" "$ROOT/corpus.bin" "$ROOT/sa.bin"
"$ROOT/build/build_bwt_binary_v1" "$ROOT/corpus.bin" "$ROOT/sa.bin" "$ROOT/bwt.bin"

N="$(stat -c %s "$ROOT/corpus.bin")"
ROWS=$((N+1))

python3 experiments/personal_vault_v0/rlb3x_fixed.py "$ROOT/bwt.bin" "$ROOT/bwt.rlb3x" "$ROOT/rlb3x-report.json" --block-runs 8192
python3 experiments/personal_vault_v0/restore_rlb3x.py "$ROOT/bwt.rlb3x" "$ROOT/restored-rlb3x.bin" "$ROOT/rlb3x-restore.json"
cmp "$ROOT/corpus.bin" "$ROOT/restored-rlb3x.bin"
python3 experiments/personal_vault_v0/vault_v0.py verify-objects "$ROOT/restored-rlb3x.bin" "$ROOT/objects.json" "$ROOT/input"

python3 experiments/personal_vault_v0/loc2_experimental.py build "$ROOT/sa.bin" "$ROWS" 128 "$ROOT/locate.loc2"
python3 experiments/personal_vault_v0/loc2_experimental.py verify "$ROOT/sa.bin" "$ROOT/locate.loc2"

python3 experiments/personal_vault_v0/vault_v0.py queries "$ROOT/corpus.bin" "$ROOT/objects.json" "$ROOT/queries.json"
python3 experiments/personal_vault_v0/vault_v0.py boundaries "$ROOT/corpus.bin" "$ROOT/objects.json" "$ROOT/boundaries.json"

python3 - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); source=(r/'corpus.bin').stat().st_size
files={'rlb3x':r/'bwt.rlb3x','loc2':r/'locate.loc2','objects':r/'objects.json'}
parts={k:p.stat().st_size for k,p in files.items()}
manifest={'format':'GLYPH_PERSONAL_VAULT_KERNEL_V0_MEASURED','version':0,'source_bytes':source,'parts':{},'runtime_bytes_without_manifest':sum(parts.values())}
for k,p in files.items():
    manifest['parts'][k]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
raw=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
(r/'kernel-v0-manifest.json').write_bytes(raw)
total=sum(parts.values())+len(raw)
report={'format':'GLYPH_PERSONAL_VAULT_KERNEL_V0_FAST','source_bytes':source,'rlb3x_bytes':parts['rlb3x'],'loc2_bytes':parts['loc2'],'object_map_bytes':parts['objects'],'manifest_bytes':len(raw),'total_runtime_bytes':total,'runtime_ratio':total/source,'direct_restore_from_rlb3x':True,'query_substrate':'RLB3X+LOC2+object-boundary-filter','rlb2_present':False,'rlr2_present':False,'frontier_probes_run':False,'important_limitation':'Rank3X still rebuilds block-prefix rank state by scanning/decompressing RLB3X blocks at process startup.'}
(r/'kernel-v0-size.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(report,sort_keys=True))
PY

test ! -e "$ROOT/bwt.rlb2"
test ! -e "$ROOT/bwt.rlr2"
echo GLYPH_PERSONAL_VAULT_KERNEL_V0_FAST_OK
