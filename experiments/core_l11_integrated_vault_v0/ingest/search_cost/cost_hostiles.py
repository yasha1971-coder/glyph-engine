"""Actual live/private/admission/format mutations; exact reasons required."""
import hashlib,json,shutil,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parent))
from session import Session
from vault import canon
from packed_storage import expand

def run(vault,head,reference,output,limits):
    output.mkdir();results=[]
    def rejection(name,fn,kind,reason):
        try:fn()
        except kind as e:
            assert reason in str(e),(name,str(e));results.append(dict(name=name,exception=type(e).__name__,reason=str(e)))
        else:raise AssertionError('accepted hostile '+name)
    def pool(name,edit,reason='live Vault changed; reopen session'):
        with tempfile.TemporaryDirectory(dir=output,prefix=name+'-') as tmp:
            path=Path(tmp)/'vault';shutil.copytree(vault,path);s=Session(path,head,reference)
            try:edit(path,s);rejection(name,lambda:s.query(head,b'never-present'),ValueError,reason)
            finally:s.close()
    obj=next((vault/'objects').glob('*/ef.packed')).parent.name
    pool('missing-active-component',lambda v,s:(v/'objects'/obj/'ef.packed').unlink())
    current=json.loads((vault/'snapshots'/head).read_bytes());current_keys={d['sha256'] for d in current['documents']}
    old=next(p.name for p in (vault/'objects').iterdir() if p.name not in current_keys)
    pool('missing-history-only-component',lambda v,s:(v/'objects'/old/'ef.packed').unlink())
    empty=hashlib.sha256(b'').hexdigest()
    pool('missing-explicit-empty',lambda v,s:(v/'objects'/empty/'empty.json').unlink())
    pool('decoder-changed',lambda v,s:(v/'decoder').write_bytes((v/'decoder').read_bytes()+b'x'))
    pool('CURRENT-changed',lambda v,s:(v/'CURRENT').write_text('0'*64+'\n'))
    pool('required-snapshot-changed',lambda v,s:(v/'snapshots'/head).write_bytes(b'{}\n'))
    pool('unknown-retained-addition',lambda v,s:(v/'unexpected').write_bytes(b'not free'))
    pool('private-EF-mutated',lambda v,s:(Path(next(iter(s.states.values())).stage.name)/'ef').write_bytes(b'bad'), 'private snapshot changed; reopen session')
    s=Session(vault,head,reference)
    stage=Path(s.stage.name)
    try:rejection('wrong-caller-pin',lambda:s.query('0'*64,b'absent'),ValueError,'caller snapshot pin mismatch')
    finally:s.close()
    assert not stage.exists()
    rejection('closed-session',lambda:s.query(head,b'absent'),ValueError,'session closed')
    for name,kwargs,reason in [('temp-admission',dict(temp_budget=1),'session temporary byte admission'),
        ('object-admission',dict(object_limit=1),'active object admission'),('source-admission',dict(source_budget=1),'active source admission')]:
        rejection(name,lambda:Session(vault,head,reference,**kwargs),ValueError,reason)
    # Same valid physical bytes/source geometry; re-pin the complete leaf/root
    # around a deliberately foreign magic. RLB2 reader must reject before query.
    with tempfile.TemporaryDirectory(dir=output,prefix='foreign-magic-') as tmp:
        tmp=Path(tmp);v=tmp/'vault';shutil.copytree(vault,v)
        first=head
        while True:
            r=json.loads((v/'snapshots'/first).read_bytes())
            if r['parent'] is None:break
            first=r['parent']
        doc=r['documents'][0];leaf=v/'objects'/doc['sha256'];expanded=tmp/'expanded';expand(leaf,expanded)
        raw=bytearray((expanded/'rlb2').read_bytes());raw[:8]=b'GLYRLB3X';(leaf/'rlb2.packed').write_bytes(raw)
        storage=json.loads((leaf/'storage.json').read_bytes());storage['rlb2']='raw';(leaf/'storage.json').write_bytes(canon(storage))
        physical=json.loads((leaf/'root.json').read_bytes());physical['artifacts']['rlb2']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        physical=canon(physical);(leaf/'root.json').write_bytes(physical);doc['manifest']=hashlib.sha256(physical).hexdigest()
        rawroot=canon(r);pin=hashlib.sha256(rawroot).hexdigest();(v/'snapshots'/pin).write_bytes(rawroot);(v/'CURRENT').write_text(pin+'\n')
        rejection('repinned-foreign-RLB3X-magic',lambda:Session(v,pin,reference),RuntimeError,'RLB2 magic mismatch')
    # A new independent session remains possible for historical scope. No cache
    # silently retargets to a different root; old cold API stays untouched.
    old_pin=current['parent'];s=Session(vault,old_pin,reference)
    try:
        assert s.query(old_pin,b'REVISION_ONLY')['coordinates']==[[0,2,0]]
    finally:s.close()
    with tempfile.TemporaryDirectory(dir=output,prefix='real-append-') as tmp:
        from vault_packed import VaultPacked
        from packed_history import restore
        tmp=Path(tmp);v=tmp/'vault';shutil.copytree(vault,v);s=Session(v,head,reference)
        try:
            before=s.query(head,b'BINARY_END_ONLY')
            raw=restore(v,s.root['documents'][0],tmp)
            source=tmp/'input';source.mkdir();raw.rename(source/'new-duplicate')
            writer=VaultPacked(v,reference);new=writer.import_files(source,[{'name':'new-duplicate'}],head,v/'decoder')
            rejection('append-invalidates-existing-session',lambda:s.query(head,b'absent'),ValueError,'live Vault changed; reopen session')
            rejection('append-cannot-silently-retarget',lambda:s.query(new,b'absent'),ValueError,'caller snapshot pin mismatch')
            assert writer.query(head,b'BINARY_END_ONLY')==before,'default old-snapshot query changed'
        finally:s.close()
        old=Session(v,head,reference)
        try:assert old.query(head,b'BINARY_END_ONLY')==before,'old snapshot cannot reopen'
        finally:old.close()
    from monitor import run as monitored
    bad=dict(limits,sampled_tree_rss_budget_bytes=1)
    rejection('sampled-RSS-budget-kills-only-owned-worker',lambda:monitored('session',vault,head,reference,
        [dict(query_hex=b'absent'.hex(),limit=256)],output/'rss-budget',bad),RuntimeError,'sampled_process_tree_rss_budget')
    (output/'result.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n');return results
