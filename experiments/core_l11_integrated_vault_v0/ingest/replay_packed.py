"""Clean-runner packed-only capsule replay; imports no Vault or producer."""
import argparse,hashlib,json,shutil,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from packed_storage import view
from independent_scale import verify
from independent_verify import roots
p=argparse.ArgumentParser()
p.add_argument('--capsule',type=Path,required=True)
p.add_argument('--pins-json',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cases=json.loads(a.pins_json);assert type(cases) is list and len(cases)==3
vault=a.capsule/'vault'
assert not (a.capsule/'source').exists()
def replay(case,claim):
 with tempfile.TemporaryDirectory() as t:
  target=Path(t);view(vault,target)
  return verify(target,case['pin'],bytes.fromhex(case['query_hex']),256,claim)
results=[];versions={}
for i,c in enumerate(cases):
 raw=(a.capsule/(str(i)+'.json')).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==c['claim_sha256'],'separately expected claim pin'
 claim=json.loads(raw);results.append(replay(c,claim))
 root=roots(vault,c['pin'],'GLYPH_BOUNDED_IMPORT_STREAM_V2',100000000)
 for d in root['documents']:versions[d['sha256']]=d['bytes']
retained=sum(p.stat().st_size for p in vault.rglob('*') if p.is_file())
assert retained<=sum(versions.values()),'complete retained budget'
negatives=[]
def rejects(label,fn):
 try:fn()
 except Exception:negatives.append(label);return
 raise AssertionError('negative accepted: '+label)
c=cases[1];claim=json.loads((a.capsule/'1.json').read_bytes())
wrong=dict(claim);wrong['count']+=1
rejects('wrong-count',lambda:replay(c,wrong))
wrongpin=dict(c);wrongpin['pin']='0'*64
rejects('wrong-external-root-pin',lambda:replay(wrongpin,claim))
component=next((vault/'objects').glob('*/ef.packed'));saved=component.read_bytes()
try:
 component.unlink();rejects('missing-component',lambda:replay(c,claim))
 component.write_bytes(saved+b'x');rejects('corrupt-component',lambda:replay(c,claim))
finally:component.write_bytes(saved)
report=dict(status='CLEAN_PACKED_CAPSULE_PASS',replays=results,retained_bytes=retained,distinct_source_bytes=sum(versions.values()),negatives_rejected=negatives,original_sources_transferred=False,independent_index_source_semantics=False)
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
