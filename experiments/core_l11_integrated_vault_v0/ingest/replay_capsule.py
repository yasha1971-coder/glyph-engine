"""Clean-runner replay driver. Stdlib only; no producer/importer imports."""
import argparse, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--capsule',type=Path,required=True)
p.add_argument('--pins-json',required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); pins=json.loads(a.pins_json)
assert set(pins)=={'cases'} and len(pins['cases'])==12
assert not a.output.exists(); a.output.mkdir(parents=True)
rows=[]
for i,c in enumerate(pins['cases']):
    assert c['claim']==str(i)+'.json'
    command=[sys.executable,str(Path(__file__).with_name('independent_verify.py')),'--vault',str(a.capsule/'vault'),'--pin',c['pin'],'--query-hex',c['query_hex'],'--limit',str(c['limit']),'--claim',str(a.capsule/c['claim']),'--reference',str(a.reference)]
    if i==8: command+=['--selected','0','--output',str(a.output/'selected.raw')]
    r=subprocess.run(command,capture_output=True,text=True,timeout=120)
    (a.output/(str(i)+'.log')).write_text(r.stdout+r.stderr)
    assert r.returncode==0,(i,r.stderr)
    rows.append(json.loads(r.stdout))
assert (a.output/'selected.raw').read_bytes()==b'revised shared'
result=dict(status='BOUNDED_INGEST_CLEAN_RUNNER_REPLAY_PASS',logical_cases=len(rows),physical_replays=sum(x['physical_replays'] for x in rows),selected_restore=True)
(a.output/'result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
