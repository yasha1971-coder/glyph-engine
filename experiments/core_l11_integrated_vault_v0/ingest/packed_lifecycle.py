"""Bounded synthetic packed lifecycle / real SIGKILL gate. One local executor."""
import argparse,copy,json,os,random,shutil,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from vault import AcceptanceRejected,sha,canon
from vault_packed import VaultPacked
from snapshot_contract import load_chain
from packed_history import authenticate
from lifecycle_costs import inventory,disk,measured


def identity(inv):return {n:(v['bytes'],v['sha256']) for n,v in inv.items()}
def preserved(before,after):
    a,b=identity(before),identity(after)
    assert all(b.get(n)==r for n,r in a.items() if n!='CURRENT'),'old retained bytes changed'

def main():
    p=argparse.ArgumentParser()
    for n in ('reference','decoder','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--worker-vault',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--parent');p.add_argument('--changes');p.add_argument('--recover',action='store_true')
    a=p.parse_args()
    if a.worker_vault:
        v=VaultPacked(a.worker_vault,a.reference)
        if a.recover:v.recover();print(json.dumps(dict(status='RESTARTED',head=v.current())));return
        try:head=v.import_files(a.source,json.loads(a.changes),a.parent,a.decoder)
        except ValueError as e:
            if str(e)=='writer parent conflict':print('WRITER_PARENT_CONFLICT');sys.exit(4)
            raise
        print(json.dumps(dict(status='WRITER_PUBLISHED',head=head)));return
    out=a.output;out.mkdir(parents=True,exist_ok=False);src=out/'source';src.mkdir()
    text=(b'GLYPH packed snapshot exact search\n'*20000)[:524288]
    binary=(bytes(range(256))*1100)[:262144-len(b'BINARY_END_ONLY')]+b'BINARY_END_ONLY'
    revised=b'REVISION_ONLY\n'+text
    crash_source=b'CRASH_ONLY\n'+(b'crash fixture common\n'*20000)[:262144-len(b'CRASH_ONLY\n')]
    for n,b in [('text',text),('binary',binary),('empty',b''),('duplicate',text),('renamed',text),('budget-dup',text),('revision-original',revised),('crash',crash_source)]:(src/n).write_bytes(b)
    v=VaultPacked(out/'vault',a.reference);pins=[];states=[];costs=[];preservations=[];receipts={}
    changes=[([{'name':'text'},{'name':'binary'},{'name':'empty'}],None),
             ([{'name':'duplicate'}],None),([{'name':'renamed','doc_id':0}],None),
             ([{'name':'renamed','doc_id':0}],revised),([{'name':'renamed','doc_id':0}],text)]
    for index,(items,data) in enumerate(changes):
        if data is not None:(src/'renamed').write_bytes(data)
        old=inventory(v.path) if v.path.exists() else {}
        head,m=measured(lambda:v.import_files(src,items,pins[-1] if pins else None,a.decoder));pins.append(head);costs.append(m)
        now=inventory(v.path);preserved(old,now)
        root=v.snapshot(head);states.append(root)
        assert len(list((v.path/'objects').iterdir()))==(3 if index<3 else 4)
        h,hm=measured(lambda:authenticate(v.path,head))
        preservations.append(dict(snapshot=head,old_files_unchanged=len(old)-int('CURRENT' in old),history=h,history_cost=hm,retained=disk(v.path)))
        assert root['documents'][0]['doc_id']==0 and root['documents'][0]['revision']==max(0,index-1)
        for q in (b'REVISION_ONLY',b'BINARY_END_ONLY',b'GLYPH'):
            claim=v.query(head,q);receipts[(head,q.hex(),256)]=canon(claim)
            expected=[(revised if index==3 else text),binary,b'']+([text] if index else [])
            hits=[[i,root['documents'][i]['revision'],j] for i,b in enumerate(expected) for j in range(len(b)) if b.startswith(q,j)]
            assert claim['count']==len(hits)
            if any(sum(b.startswith(q,j) for j in range(len(b)))>256 for b in expected):
                assert claim['enumeration']=='OMITTED_HIGH_COUNT' and claim['status']=='UNTRUSTED' and claim['coordinates']==[]
            else:assert claim['coordinates']==hits
    assert states[1]['documents'][0]['manifest']==states[1]['documents'][3]['manifest']
    assert states[2]['documents'][0]['sha256']==states[0]['documents'][0]['sha256']
    assert states[4]['documents'][0]['sha256']==states[0]['documents'][0]['sha256']
    for (pin,q,limit),raw in receipts.items():assert canon(v.query(pin,bytes.fromhex(q),limit))==raw,'old receipt changed'
    # Runtime negatives assert precise reason and exact old file identities.
    negatives=[]
    def reject(name,fn,kind,reason,store=v):
        before=identity(inventory(store.path))
        try:fn()
        except kind as e:
            assert reason in str(e),(name,type(e).__name__,str(e));negatives.append(dict(name=name,reason=str(e)))
        else:raise AssertionError('negative accepted: '+name)
        assert identity(inventory(store.path))==before,(name,'rejection changed retained bytes')
    head=pins[-1]
    reject('stale-parent',lambda:v.import_files(src,[{'name':'budget-dup'}],pins[0],a.decoder),ValueError,'writer parent conflict')
    reject('duplicate-path',lambda:v.import_files(src,[{'name':'duplicate'}],head,a.decoder),ValueError,'document/path bound')
    for ident in (True,99,-1):reject('invalid-doc-id-'+str(ident),lambda:v.import_files(src,[{'name':'renamed','doc_id':ident}],head,a.decoder),ValueError,'explicit stable identity required')
    for label,edit,reason in [('bad-revision',lambda r:r['documents'][0].update(revision=999),'revision lineage'),('bad-lineage',lambda r:r.update(vault_id='0'*32),'vault lineage')]:
        bad=copy.deepcopy(states[-1]);edit(bad);raw=canon(bad);pin=sha(raw);file=v.path/'snapshots'/pin;file.write_bytes(raw)
        try:reject(label,lambda:load_chain(v.path,pin,v.FORMAT,v.SOURCE_LIMIT),ValueError,reason)
        finally:file.unlink()
    # Unknown retained bytes consume budget. A duplicate adds no denominator.
    def exhaust(store):
        h=authenticate(store.path,store.current());side=store.path/'required-history-sidecar'
        side.write_bytes(b'\0'*(h['distinct_source_bytes']-h['retained_bytes']));return side
    reserve=exhaust(v)
    reject('history-sidecars-exhaust-budget',lambda:v.import_files(src,[{'name':'budget-dup'}],head,a.decoder),AcceptanceRejected,'full retained storage budget exceeded')
    reserve.unlink()
    # Missing historical components may not fund another accepted snapshot.
    historical=out/'missing-old-component';shutil.copytree(v.path,historical)
    missing=states[3]['documents'][0]['sha256']
    (historical/'objects'/missing/'ef.packed').unlink()
    damaged=VaultPacked(historical,a.reference)
    reject('writer-missing-old-version-component',lambda:damaged.import_files(src,[{'name':'budget-dup'}],head,a.decoder),FileNotFoundError,missing+'/ef.packed',damaged)
    foreign=out/'unknown-staging';shutil.copytree(v.path,foreign)
    unknown=foreign/'staging/foreign';unknown.mkdir();(unknown/'payload').write_bytes(b'UNOWNED retained bytes')
    known=identity(inventory(foreign));VaultPacked(foreign,a.reference).recover()
    assert identity(inventory(foreign))==known,'recovery deleted unowned staging'
    # A valid legacy orphan may be reused, but is not owned by this transaction.
    # Publication target and deletion ownership must remain separate.
    legacy=out/'preexisting-snapshot';shutil.copytree(v.path,legacy)
    pending=VaultPacked(legacy,a.reference)
    prepared=pending.import_files(src,[{'name':'crash'}],head,a.decoder)
    (legacy/'CURRENT').write_text(head+'\n') # isolated legacy-orphan simulation
    preexisting=inventory(legacy)
    reused=pending.import_files(src,[{'name':'crash'}],head,a.decoder)
    assert reused==prepared and pending.current()==prepared
    preserved(preexisting,inventory(legacy));authenticate(legacy,reused)
    (src/'random').write_bytes(random.Random(20261109).randbytes(65536));(src/'tiny').write_bytes(b'x')
    for n in ('random','tiny'):
        bad=VaultPacked(out/('rejected-'+n),a.reference)
        try:bad.import_files(src,[{'name':n}],None,a.decoder)
        except AcceptanceRejected as e:assert 'full retained storage budget exceeded' in str(e);negatives.append(dict(name='initial-'+n,reason=str(e)))
        else:raise AssertionError('incompressible/tiny compact acceptance')
        assert bad.current() is None and not list((bad.path/'objects').iterdir()) and not list((bad.path/'snapshots').iterdir())
        assert (src/n).exists(),'original synthetic source must survive rejection'
    def worker(path,parent,change=None,recover=False):
        cmd=[sys.executable,'-I',str(HERE/'packed_lifecycle.py'),'--reference',str(a.reference),'--decoder',str(a.decoder),'--output',str(out),'--worker-vault',str(path)]
        if recover:cmd+=['--recover']
        else:cmd+=['--source',str(src),'--parent',parent,'--changes',json.dumps(change)]
        return cmd
    concurrent=out/'concurrent';shutil.copytree(v.path,concurrent)
    command=worker(concurrent,head,[{'name':'crash'}]);children=[subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
    writers=[]
    for child in children:
        stdout,stderr=child.communicate(timeout=180);writers.append(dict(exit=child.returncode,stdout=stdout.decode(),stderr=stderr.decode()))
    assert sorted(w['exit'] for w in writers)==[0,4],writers
    assert 'WRITER_PARENT_CONFLICT' in next(w['stdout'] for w in writers if w['exit']==4)
    assert VaultPacked(concurrent,a.reference).query(VaultPacked(concurrent,a.reference).current(),b'CRASH_ONLY')['count']==1
    crashes=[]
    for point in ('object-ready','snapshot-ready','before-current','after-current','published'):
        path=out/('crash-'+point);shutil.copytree(v.path,path);old=inventory(path)
        child=subprocess.run(worker(path,head,[{'name':'crash'}]),capture_output=True,text=True,env={**os.environ,'GLYPH_TEST_CRASH':point},timeout=180)
        assert child.returncode==-9,(point,child.returncode,child.stderr)
        killed=disk(path)
        restarted=subprocess.run(worker(path,head,recover=True),capture_output=True,text=True,timeout=60)
        assert restarted.returncode==0,(point,restarted.stderr)
        store=VaultPacked(path,a.reference);at_restart=store.current();preserved(old,inventory(path))
        assert (at_restart==head)==(point in ('object-ready','snapshot-ready','before-current'))
        assert not list((path/'staging').iterdir()),'owned crash staging not recovered'
        h=authenticate(path,at_restart)
        assert store.query(head,b'REVISION_ONLY')['count']==0
        assert store.query(at_restart,b'CRASH_ONLY')['count']==(0 if at_restart==head else 1)
        side=exhaust(store)
        reject('crash-budget-rejection-'+point,lambda:store.import_files(src,[{'name':'budget-dup'}],at_restart,a.decoder),AcceptanceRejected,'full retained storage budget exceeded',store)
        side.unlink()
        final=store.import_files(src,[{'name':'crash'}] if at_restart==head else [{'name':'budget-dup'}],at_restart,a.decoder)
        proof=authenticate(path,final);assert store.query(final,b'CRASH_ONLY')['count']==1
        preserved(old,inventory(path))
        crashes.append(dict(point=point,exit=child.returncode,restart_exit=restarted.returncode,head_at_restart=at_restart,
            before_recovery=killed,after_recovery=h,final=proof,final_retained=disk(path)))
    # Only immutable packed Vault and bounded query claims are exported.
    capsule=out/'capsule';capsule.mkdir();shutil.copytree(v.path,capsule/'vault');cases=[]
    boundary=text[-8:]+binary[:8]
    for pin,q,limit in [*receipts.keys(),(head,boundary.hex(),256),(head,b'BINARY_END_ONLY'.hex(),0)]:
        data=receipts.get((pin,q,limit)) or canon(v.query(pin,bytes.fromhex(q),limit))
        name=str(len(cases))+'.json';(capsule/name).write_bytes(data)
        cases.append(dict(pin=pin,query_hex=q,limit=limit,claim_sha256=sha(data)))
    assert json.loads((capsule/(str(len(cases)-2)+'.json')).read_bytes())['count']==0,'cross-document trap was a real document match'
    selected=[]
    for i,doc in ((0,0),(3,0),(4,0),(0,1),(0,2)):
        d=states[i]['documents'][doc];original=binary if doc==1 else b'' if doc==2 else revised if i==3 else text
        assert d['sha256']==sha(original) and d['bytes']==len(original)
        selected.append(dict(pin=pins[i],doc_id=doc,revision=d['revision'],sha256=sha(original),bytes=len(original)))
    expected=dict(format='GLYPH_PACKED_LIFECYCLE_PINS_V1',head=head,cases=cases,selected=selected)
    (out/'pins.json').write_bytes(canon(expected))
    report=dict(status='PACKED_LIFECYCLE_LOCAL_PASS',head=head,snapshots=pins,query_cases=len(cases),old_receipts_unchanged=len(receipts),
        independent_index_source_semantics=False,import_costs=costs,snapshot_evidence=preservations,
        negatives=negatives,concurrent_writers=writers,crashes=crashes,retained=disk(v.path),
        preexisting_snapshot_reused_without_cleanup_error=True,
        synthetic_seed=20261109,current_logical_bytes=sum(d['bytes'] for d in states[-1]['documents']),
        source_original_files_preserved=True,power_loss_proven=False,
        scratch_file_lengths=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    report['checkout_head']=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip()
    report['checkout_tree']=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD^{tree}'],text=True).strip()
    report['code_sha256']={str(p.relative_to(HERE.parents[2])):sha(p.read_bytes()) for p in HERE.glob('*.py')}
    return out,report

if __name__=='__main__':
    result,cost=measured(main)
    if result is not None:
        out,report=result;report['whole_gate_cost']=cost
        (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        print(json.dumps(report,sort_keys=True),flush=True);print('PACKED_LIFECYCLE_PRODUCER_PASS',flush=True)
