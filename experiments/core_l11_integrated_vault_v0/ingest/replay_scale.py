"""Clean-runner replay of separately pinned scaled cases; no source inputs."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capsule',type=Path,required=True);p.add_argument('--pins-json',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True);cases=json.loads(a.pins_json);assert len(cases)==3
for i,c in enumerate(cases):
 cmd=[sys.executable,'-I',str(Path(__file__).with_name('independent_scale.py')),'--vault',str(a.capsule/'vault'),'--pin',c['pin'],'--query-hex',c['query_hex'],'--limit','256','--claim',str(a.capsule/(str(i)+'.json'))]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=600);(a.output/(str(i)+'.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,(i,r.stdout,r.stderr)
print('CLEAN_SCALED_VAULT_3_REPLAYS_PASS')
