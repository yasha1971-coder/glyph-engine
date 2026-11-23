"""Isolated real-file 4K full-loop producer and clean-capsule replay gate."""
import argparse,copy,json,os,random,shutil,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

def invoke(capsule,case,output,claim=None):
    file=capsule/case['claim']
    if claim is not None:
        file=output/'hostile.json';file.write_text(json.dumps(claim,sort_keys=True,separators=(',',':'))+'\n')
    cmd=[sys.executable,'-I',str(HERE/'run_independent_4k.py'),'--vault',str(capsule/'vault'),'--pin',case['pin'],'--query-hex',case['query_hex'],'--limit',str(case['limit']),'--claim',str(file)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    with (output/'commands.jsonl').open('a') as log:log.write(json.dumps(dict(command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))+'\n')
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['produce','replay'],required=True)
    for n in ('output','capsule'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--decoder',type=Path);p.add_argument('--pins-json');p.add_argument('--worker',nargs=3)
    a=p.parse_args()
    if a.worker:
        from vault_4k import Vault4K
        Vault4K(a.worker[0],a.reference).import_files(Path(a.worker[1]),[{'name':'new.bin'}],a.worker[2],a.decoder);return
    assert not a.output.exists();a.output.mkdir(parents=True)
    if a.phase=='replay':
        pins=json.loads(a.pins_json);assert len(pins['cases'])==12
        for case in pins['cases']:
            r=invoke(a.capsule,case,a.output);assert r.returncode==0,(r.stdout,r.stderr)
        report=dict(status='CLEAN_RUNNER_4K_VAULT_PASS',logical_cases=12)
    else:
        from vault_4k import Vault4K
        from vault import Vault
        rng=random.Random(4096)
        with tempfile.TemporaryDirectory(prefix='glyph-4k-vault-') as tmp:
            tmp=Path(tmp);src=tmp/'source';src.mkdir()
            raw=b'x'*600+bytes(rng.randrange(128,256) for _ in range(3496))
            (src/'large.bin').write_bytes(raw);(src/'empty').write_bytes(b'')
            v=Vault4K(tmp/'vault',a.reference)
            first=v.import_files(src,[{'name':'large.bin'},{'name':'empty'}],None,a.decoder)
            (src/'a-copy.bin').write_bytes(raw)
            second=v.import_files(src,[{'name':'a-copy.bin'}],first,a.decoder)
            (src/'revision.bin').write_bytes(b'NEW_REV'+raw[7:])
            third=v.import_files(src,[{'name':'revision.bin','doc_id':0}],second,a.decoder)
            try:Vault(v.path,a.reference).snapshot(third)
            except ValueError:pass
            else:raise AssertionError('V0 accepted V1')
            a.capsule.mkdir(parents=True);shutil.copytree(v.path,a.capsule/'vault')
            cases=[]
            for pin in (first,second,third):
                for pattern,limit in ((b'x',256),(b'xx',0),(b'absent-4k-query',256),(b'NEW_REV',1)):
                    claim=v.query(pin,pattern,limit)
                    if pattern==b'x':assert claim['count']>256 and len(claim['coordinates'])==256 and claim['status']=='UNTRUSTED'
                    name=str(len(cases))+'.json';(a.capsule/name).write_text(json.dumps(claim,sort_keys=True,separators=(',',':'))+'\n')
                    cases.append(dict(pin=pin,query_hex=pattern.hex(),limit=limit,claim=name))
            for case in cases:
                r=invoke(a.capsule,case,a.output);assert r.returncode==0,(r.stdout,r.stderr)
            original=json.loads((a.capsule/'8.json').read_text());bad=0
            for mutate in (lambda x:x.update(count=0),lambda x:x.update(status='FOUND'),lambda x:x.update(enumeration_complete=True),lambda x:x.update(covered_doc_ids=[]),lambda x:x.update(coordinates=[])):
                fake=copy.deepcopy(original);mutate(fake);assert invoke(a.capsule,cases[8],a.output,fake).returncode==3;bad+=1
            doc=v.snapshot(third)['documents'][0]
            ef=a.capsule/'vault'/'objects'/doc['sha256']/'ef'
            saved=ef.read_bytes(); ef.unlink()
            assert invoke(a.capsule,cases[10],a.output).returncode==3;bad+=1
            ef.write_bytes(b'corrupt')
            assert invoke(a.capsule,cases[10],a.output).returncode==3;bad+=1
            ef.write_bytes(saved)
            # Process-crash/restart uses inherited durable implementation with V1 records.
            (src/'new.bin').write_bytes(raw);kills=0
            for point in ('object-ready','snapshot-ready','before-current','after-current','published'):
                # Unique content forces object construction for object-ready.
                (src/'new.bin').write_bytes(b'NEW'+raw[3:])
                path=tmp/point;shutil.copytree(v.path,path)
                cmd=[sys.executable,str(HERE/'gate_4k.py'),'--phase','produce','--output',str(a.output),'--capsule',str(a.capsule),'--reference',str(a.reference),'--decoder',str(a.decoder),'--worker',str(path),str(src),third]
                result=subprocess.run(cmd,env=dict(os.environ,GLYPH_TEST_CRASH=point),capture_output=True)
                assert result.returncode==-9,(point,result.stderr)
                fresh=Vault4K(path,a.reference);current=fresh.current();assert fresh.query(current,b'NEW',1)['count'] in (1,2)
                if current==third:fresh.import_files(src,[{'name':'new.bin'}],third,a.decoder)
                assert fresh.query(fresh.current(),b'NEW',1)['count']==2;kills+=1
            (src/'too-large').write_bytes(raw+b'!')
            try:v.import_files(src,[{'name':'too-large'}],third,a.decoder)
            except ValueError:pass
            else:raise AssertionError('4097-byte input accepted')
            assert v.current()==third
            pins=dict(cases=cases);(a.output/'pins.json').write_text(json.dumps(pins,sort_keys=True)+'\n')
            report=dict(status='REAL_FILE_4K_VAULT_PRODUCER_PASS',snapshots=3,logical_cases=12,capped_prefix=256,initial_high_count=600,hostiles_rejected=bad,sigkill_boundaries=kills,oversize_rejected=True,original_v0_rejects_v1=True)
    (a.output/'result.json').write_text(json.dumps(report,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
