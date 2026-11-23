"""Bounded aggregate growth; synthetic distinct contents, explicit time budget."""
import argparse,hashlib,json,random,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from vault_scale import VaultScale
p=argparse.ArgumentParser()
for n in ('reference','decoder','output'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--count',type=int,default=10);p.add_argument('--bytes',type=int,default=100000000);p.add_argument('--budget-seconds',type=int,default=1200);a=p.parse_args()
assert 1<=a.count<=64 and 4096<a.bytes<=100000000 and not a.output.exists()
a.output.mkdir(parents=True);src=a.output/'sources';src.mkdir();v=VaultScale(a.output/'vault',a.reference);pin=None;rows=[];started=time.monotonic()
for index in range(a.count):
 if index and time.monotonic()-started> a.budget_seconds-180:break
 before=time.monotonic();rng=random.Random(827000+index)
 block=bytes(rng.randrange(256) for _ in range(65536));tag=('GLYPH_AGGREGATE_%02d_END'%index).encode();raw=(tag+block*((a.bytes+65535)//65536))[:a.bytes]
 file=src/('%02d.bin'%index);file.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest();del raw
 old=pin;pin=v.import_files(src,[{'name':file.name}],pin,a.decoder)
 assert len(v.snapshot(pin)['documents'])==index+1
 if old:assert len(v.snapshot(old)['documents'])==index
 u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
 row=dict(document=index,bytes=a.bytes,source_sha256=digest,snapshot=pin,wall_seconds=time.monotonic()-before,self_peak_rss_kib=u.ru_maxrss,max_child_rss_kib=c.ru_maxrss)
 rows.append(row);(a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
assert rows
needle=b'GLYPH_AGGREGATE_';claim=v.query(pin,needle,256);assert claim['count']==len(rows)
file=a.output/'claim.json';file.write_text(json.dumps(claim,sort_keys=True,separators=(',',':'))+'\n')
r=subprocess.run([sys.executable,'-I',str(HERE/'independent_scale.py'),'--vault',str(v.path),'--pin',pin,'--query-hex',needle.hex(),'--claim',str(file)],capture_output=True,text=True,timeout=600)
(a.output/'independent.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(r.stdout,r.stderr)
assert len({x['source_sha256'] for x in rows})==len(rows)
report=dict(status='AGGREGATE_FULL_REPLAY_PASS',distinct_files=len(rows),source_logical_bytes=len(rows)*a.bytes,snapshots=len(rows),physical_objects=len(list((v.path/'objects').iterdir())),retained_vault_bytes=sum(p.stat().st_size for p in v.path.rglob('*') if p.is_file()),wall_seconds=time.monotonic()-started,protocol_file_cap=100000000,hardware_maximum_established=False,independent_index_source_semantics=False,stopped_at_target=len(rows)==a.count)
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
