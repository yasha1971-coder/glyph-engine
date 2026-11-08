import argparse,json,os,random,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from vault_packed import VaultPacked
from vault import AcceptanceRejected,sha
from packed_storage import view
from independent_scale import verify
p=argparse.ArgumentParser()
for n in ('reference','decoder','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);src=a.output/'source';src.mkdir()
v=VaultPacked(a.output/'vault',a.reference);started=time.monotonic()
raw=(b'GLYPH packed snapshot exact search\n'*40000)[:1<<20]
(src/'a').write_bytes(raw);pin=v.import_files(src,[{'name':'a'}],None,a.decoder)
before={str(p.relative_to(v.path)):sha(p.read_bytes()) for p in v.path.rglob('*') if p.is_file()}
(src/'b').write_bytes(b'new version marker\n'+raw)
# Force the budget rejection path with an honest source budget failure on a
# separate incompressible candidate, and verify initial head stays unpublished.
bad=VaultPacked(a.output/'rejected',a.reference)
(src/'random').write_bytes(random.Random(123).randbytes(1<<20))
rejected=False
try: bad.import_files(src,[{'name':'random'}],None,a.decoder)
except AcceptanceRejected: rejected=True
assert rejected and bad.current() is None and not list((bad.path/'objects').iterdir()) and not list((bad.path/'snapshots').iterdir())
(src/'tiny').write_bytes(b'x')
# Exhaust remaining budget with an explicit retained auxiliary component.
# A random append can fit inside earlier savings, so is not a rejection oracle.
reserve=v.path/'required-sidecar-test'
reserve.write_bytes(b'\0'*(len(raw)-sum(p.stat().st_size for p in v.path.rglob('*') if p.is_file())))
reserved={str(p.relative_to(v.path)):sha(p.read_bytes()) for p in v.path.rglob('*') if p.is_file()}
try:
 v.import_files(src,[{'name':'tiny'}],pin,a.decoder)
 raise AssertionError('oversized append accepted')
except AcceptanceRejected: pass
assert v.current()==pin
assert {str(p.relative_to(v.path)):sha(p.read_bytes()) for p in v.path.rglob('*') if p.is_file()}==reserved
reserve.unlink()
# Same parent survives a real SIGKILL immediately before head publication.
cmd=[sys.executable,'-c',"import sys;sys.path.insert(0,sys.argv[1]);from vault_packed import VaultPacked;from pathlib import Path;VaultPacked(Path(sys.argv[2]),Path(sys.argv[3])).import_files(Path(sys.argv[4]),[{'name':'b'}],sys.argv[5],Path(sys.argv[6]))",str(HERE),str(v.path),str(a.reference),str(src),pin,str(a.decoder)]
r=subprocess.run(cmd,env={**os.environ,'GLYPH_TEST_CRASH':'before-current'})
assert r.returncode==-9 and v.current()==pin
new=v.import_files(src,[{'name':'b'}],pin,a.decoder)
assert len(v.snapshot(pin)['documents'])==1 and len(v.snapshot(new)['documents'])==2
assert all(sha((v.path/n).read_bytes())==h for n,h in before.items() if n!='CURRENT')
claim=v.query(new,b'new version marker')
with tempfile.TemporaryDirectory() as t:
 expanded=Path(t);view(v.path,expanded)
 proof=verify(expanded,new,b'new version marker',256,claim)
# Corrupt a packed component: never FOUND/PROVEN_EMPTY.
packed=next((v.path/'objects').glob('*/ef.packed'));saved=packed.read_bytes();packed.write_bytes(saved+b'x')
try:
 try: v.query(new,b'absent');raise AssertionError('corruption accepted')
 except (ValueError,EOFError): pass
finally: packed.write_bytes(saved)
retained=sum(p.stat().st_size for p in v.path.rglob('*') if p.is_file())
assert retained<=len(raw)*2+len(b'new version marker\n')
report=dict(status='PACKED_FULL_SNAPSHOT_PASS',retained_bytes=retained,distinct_source_bytes=len(raw)*2+len(b'new version marker\n'),snapshots=2,independent=proof,crash_restart=True,incompressible_rejected=rejected,wall_seconds=time.monotonic()-started,limits='temporary expanded indexes; full source scan verification; not constant RAM, production or real VIKA')
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
