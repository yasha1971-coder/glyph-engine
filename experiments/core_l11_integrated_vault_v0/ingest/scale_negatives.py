"""Adversarial checks against an already-created synthetic ladder stage."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('stage',type=Path);a=p.parse_args()
folder=a.stage;vault=folder/'vault';claim=json.loads((folder/'0.json').read_text());script=Path(__file__).with_name('independent_scale.py')
def check(data):
 path=folder/'forged.json';path.write_text(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n')
 result=subprocess.run([sys.executable,'-I',str(script),'--vault',str(vault),'--pin',claim['snapshot'],'--query-hex',claim['query_hex'],'--claim',str(path)],capture_output=True,text=True,timeout=600)
 assert result.returncode==3,(result.stdout,result.stderr)
for key,value in [('count',0),('independent_index_source_semantics',True),('covered_doc_ids',[])]:
 data=dict(claim);data[key]=value;check(data)
root=json.loads((vault/'snapshots'/claim['snapshot']).read_text());ef=vault/'objects'/root['documents'][0]['sha256']/'ef';saved=ef.read_bytes()
try:
 ef.unlink();check(claim);ef.write_bytes(b'bad');check(claim)
finally:ef.write_bytes(saved)
print('STREAM_NEGATIVES_5_REJECTED')
