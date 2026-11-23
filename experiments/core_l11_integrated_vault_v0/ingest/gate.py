"""Synthetic file integration and process-crash gate; no private inputs."""
import json, os, shutil, subprocess, sys, tempfile, resource, time
from pathlib import Path
from vault import Vault, sha
HERE=Path(__file__).resolve().parent
import argparse, copy
parser=argparse.ArgumentParser()
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--decoder',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--worker',nargs=3)
started=time.monotonic()
args=parser.parse_args()
ref=args.reference.resolve(); decoder=args.decoder.resolve(); output=args.output.resolve()
if args.worker:
    v=Vault(args.worker[0],ref)
    v.import_files(Path(args.worker[1]),[{'name':'new.bin'}],args.worker[2],decoder)
    raise SystemExit
assert not output.exists(), 'refuse existing evidence directory'
output.mkdir(parents=True)
from native_builder_gate import run as native_gate
native_gate(ref,decoder,output/'native-builder.json')
with tempfile.TemporaryDirectory(prefix='glyph-ingest-synthetic-') as tmp:
    base=Path(tmp); src=base/'source'; src.mkdir()
    (src/'z.bin').write_bytes(b'alpha\x00\xff shared'); (src/'empty').write_bytes(b'')
    v=Vault(base/'vault',ref)
    a=v.import_files(src,[{'name':'z.bin'},{'name':'empty'}],None,decoder)
    assert v.query(a,b'shared')['coordinates']==[[0,0,8]]
    (src/'a.bin').write_bytes((src/'z.bin').read_bytes())
    b=v.import_files(src,[{'name':'a.bin'}],a,decoder)
    assert v.query(b,b'shared')['count']==2
    assert v.query(a,b'shared')['count']==1
    assert len(list((v.path/'objects').iterdir()))==2
    (src/'rename.bin').write_bytes(b'revised shared')
    c=v.import_files(src,[{'name':'rename.bin','doc_id':0}],b,decoder)
    assert v.query(c,b'revised')['coordinates']==[[0,1,0]]
    assert v.query(b,b'revised')['status']=='PROVEN_EMPTY'
    (src/'new.bin').write_bytes(b'new shared')
    kills=0
    for point in ('object-ready','snapshot-ready','before-current','after-current','published'):
        case=base/point; shutil.copytree(v.path,case)
        env=dict(os.environ,GLYPH_TEST_CRASH=point)
        p=subprocess.run([sys.executable,str(HERE/'gate.py'),'--reference',str(ref),'--decoder',str(decoder),'--output',str(output),'--worker',str(case),str(src),c],env=env)
        assert p.returncode==-9,(point,p.returncode)
        resumed=Vault(case,ref); pin=resumed.current()
        assert resumed.query(pin,b'shared')['count'] in (2,3)
        assert resumed.query(c,b'revised')['count']==1
        if pin==c:
            pin=resumed.import_files(src,[{'name':'new.bin'}],c,decoder)
        assert resumed.query(pin,b'new')['count']==1
        kills+=1
    # Persist index-only capsule and claims before negative tests.
    capsule=output/'capsule'; capsule.mkdir()
    shutil.copytree(v.path,capsule/'vault')
    cases=[]
    for pin in (a,b,c):
        for pattern,limit in ((b'shared',256),(b'absent',256),(b'\x00\xff',256),(b'shared',0)):
            name=str(len(cases))+'.json'; claim=v.query(pin,pattern,limit)
            (capsule/name).write_text(json.dumps(claim,sort_keys=True,separators=(',',':'))+'\n')
            cases.append(dict(pin=pin,query_hex=pattern.hex(),limit=limit,claim=name))
    pins=dict(cases=cases)
    (output/'pins.json').write_text(json.dumps(pins,sort_keys=True)+'\n')
    # Fresh Python processes; verifier imports no producer-side modules.
    def independent(case,claim=None,expect=0,folder=None):
        path=capsule/case['claim']
        if claim is not None:
            path=output/'hostile.json'; path.write_text(json.dumps(claim,sort_keys=True,separators=(',',':'))+'\n')
        cmd=[sys.executable,str(HERE/'independent_verify.py'),'--vault',str(folder or capsule/'vault'),'--pin',case['pin'],'--query-hex',case['query_hex'],'--limit',str(case['limit']),'--claim',str(path),'--reference',str(ref)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        with (output/'independent.log').open('a') as log: log.write(json.dumps(dict(command=cmd,exit=result.returncode,stdout=result.stdout,stderr=result.stderr))+'\n')
        assert result.returncode==expect,(cmd,result.stdout,result.stderr)
    for case in cases: independent(case)
    hostiles=0
    original=json.loads((capsule/cases[-4]['claim']).read_text())
    for mutate in (lambda x:x.update(count=999),lambda x:x.update(coordinates=[]),lambda x:x.update(status='PROVEN_EMPTY'),lambda x:x.update(snapshot='0'*64),lambda x:x.update(query_hex='00'),lambda x:x.update(limit=1),lambda x:x.update(physical_receipts=[])):
        forged=copy.deepcopy(original); mutate(forged); independent(cases[-4],forged,3); hostiles+=1
    from independent_verify import roots
    from vault import canon
    from snapshot_contract import load_chain
    root=v.snapshot(c)
    for mutate in (lambda x:x['documents'][0].update(doc_id=True),lambda x:x['documents'][0].update(path_hex='2e2e'),lambda x:x.update(vault_id='0'*32),lambda x:x['documents'][0].update(revision=99),lambda x:x['documents'].pop(),lambda x:x.update(extra=1)):
        forged=copy.deepcopy(root); mutate(forged); raw=canon(forged); wrong=sha(raw)
        (v.path/'snapshots'/wrong).write_bytes(raw)
        for validator in (load_chain,roots):
            try: validator(v.path,wrong)
            except (ValueError,KeyError): pass
            else: raise AssertionError('repinned malformed root accepted')
        hostiles+=1
    # A missing required leaf must fail even when the expected answer is empty.
    absent=cases[-3]
    leaf=capsule/'vault'/'objects'/root['documents'][0]['sha256']/'ef'
    saved_leaf=leaf.read_bytes(); leaf.unlink(); independent(absent,expect=3); leaf.write_bytes(saved_leaf); hostiles+=1
    # Parent CAS rejects a stale writer without advancing CURRENT.
    try: v.import_files(src,[{'name':'new.bin'}],a,decoder)
    except ValueError: pass
    else: raise AssertionError('stale writer published')
    assert v.current()==c
    concurrent=base/'concurrent'; shutil.copytree(v.path,concurrent)
    worker=[sys.executable,str(HERE/'gate.py'),'--reference',str(ref),'--decoder',str(decoder),'--output',str(output),'--worker',str(concurrent),str(src),c]
    processes=[subprocess.Popen(worker,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
    outcomes=[(p.communicate(),p.returncode) for p in processes]
    assert sorted(x[1] for x in outcomes)==[0,1],outcomes
    assert Vault(concurrent,ref).query(Vault(concurrent,ref).current(),b'new')['count']==1
    (output/'writers.json').write_text(json.dumps([{'exit':x[1],'stdout':x[0][0].decode(),'stderr':x[0][1].decode()} for x in outcomes]))
    (src/'oversize').write_bytes(b'x'*257)
    (src/'link').symlink_to(src/'z.bin')
    for name in ('oversize','link','../outside','a.bin'):
        try: v.import_files(src,[{'name':name}],c,decoder)
        except (ValueError,OSError): pass
        else: raise AssertionError(('unsafe/duplicate path admitted',name))
        assert v.current()==c

    doc=v.snapshot(c)['documents'][0]
    v.restore_object(v.path/'objects'/doc['sha256'],doc['sha256'],doc['bytes'],base/'restored')
    assert (base/'restored').read_bytes()==(src/'rename.bin').read_bytes()
    obj=v.path/'objects'/doc['sha256']; saved=(obj/'ef').read_bytes(); (obj/'ef').write_bytes(b'bad')
    try: v.query(c,b'absent')
    except ValueError: pass
    else: raise AssertionError('false empty on corrupted leaf')
    (obj/'ef').write_bytes(saved)
    report=dict(status='LOCAL_BOUNDED_INGEST_SMOKE_PASS',snapshots=3,sigkill_boundaries=kills,independent_selected_restore=True,corrupt_leaf_absence_rejected=True,scope='256-byte files; local process crashes only; not full acceptance')
    report.update(independent_cases=len(cases),hostiles_rejected=hostiles,concurrent_writer_exits=[0,1],source_rejections=4)
    usage=resource.getrusage(resource.RUSAGE_SELF); child=resource.getrusage(resource.RUSAGE_CHILDREN)
    report.update(wall_seconds=time.monotonic()-started,self_cpu_seconds=usage.ru_utime+usage.ru_stime,terminated_children_cpu_seconds=child.ru_utime+child.ru_stime,self_peak_rss_kib=usage.ru_maxrss,max_child_rss_kib=child.ru_maxrss,retained_vault_logical_bytes=sum(p.stat().st_size for p in (capsule/'vault').rglob('*') if p.is_file()),cost_scope='gate processes; compiler and installed Python/stdlib/Git excluded; RSS not simultaneous process-tree peak')
    (output/'result.json').write_text(json.dumps(report,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))
