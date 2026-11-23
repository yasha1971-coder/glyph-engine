"""Independent hostile history/codec/claim tests; intended reasons mandatory."""
import copy,json,shutil,tempfile,zlib
from pathlib import Path
from independent_verify import roots,read,parse,canon,sha,demand
from packed_history import authenticate,history
from packed_storage import expand

def run(vault,head,case,claim,replay,output):
    output.mkdir();records=[]
    def rejects(name,fn,kind,expected):
        try:fn()
        except kind as e:
            demand(expected(e),'wrong rejection reason '+name+': '+str(e))
            records.append(dict(name=name,exception=type(e).__name__,reason=str(e)))
        else:raise AssertionError('hostile accepted: '+name)
    def pool(name,edit,kind,expected,operation=None):
        with tempfile.TemporaryDirectory(dir=output,prefix=name+'-') as t:
            target=Path(t)/'vault';shutil.copytree(vault,target)
            edit(target)
            rejects(name,lambda:(operation or (lambda v:authenticate(v,head)))(target),kind,expected)
    root,chain,versions=history(vault,head)
    parent=root['parent']
    pool('missing-parent',lambda v:(v/'snapshots'/parent).unlink(),FileNotFoundError,lambda e:Path(e.filename).name==parent)
    current_versions={d['sha256'] for d in root['documents']}
    old=next(d for key,d in versions.items() if key not in current_versions and d['bytes'])
    pool('missing-ancestor-only-source-object',lambda v:shutil.rmtree(v/'objects'/old['sha256']),ValueError,lambda e:str(e)=='missing historical source object')
    pool('missing-old-version-component',lambda v:(v/'objects'/old['sha256']/'ef.packed').unlink(),FileNotFoundError,lambda e:Path(e.filename).name=='ef.packed')
    pool('changed-decoder',lambda v:(v/'decoder').write_bytes((v/'decoder').read_bytes()+b'x'),ValueError,lambda e:str(e)=='history decoder identity')
    rejects('wrong-externally-transported-head',lambda:authenticate(vault,'0'*64),ValueError,lambda e:str(e)=='externally expected CURRENT')
    # All query roots may be the CURRENT root: ancestor restoration is still
    # mandatory and cannot be inferred from the selected query coverage.
    for name,edit in [('wrong-coverage',lambda c:c['covered_doc_ids'].pop()),
        ('false-empty',lambda c:c.update(status='PROVEN_EMPTY',count=0,coordinates=[])),
        ('stronger-index-semantics-claim',lambda c:c.update(independent_index_source_semantics=True)),
        ('fabricated-global-coordinate',lambda c:c.update(coordinates=[[999,0,0]],enumeration='COMPLETE'))]:
        bad=copy.deepcopy(claim);edit(bad)
        rejects(name,lambda:replay(vault,case,bad),ValueError,lambda e:str(e)=='independent answer mismatch')
    for name,edit,reason in [
        ('duplicate-path',lambda r:r['documents'][-1].update(path_hex=r['documents'][0]['path_hex']),'path identity'),
        ('bool-doc-ID',lambda r:r['documents'][0].update(doc_id=True),'document types'),
        ('revision-jump',lambda r:r['documents'][0].update(revision=999),'revision progression'),
        ('removed-logical-slot',lambda r:r['documents'].pop(),'lineage identity'),
        ('changed-vault-lineage',lambda r:r.update(vault_id='0'*32),'lineage identity')]:
        with tempfile.TemporaryDirectory(dir=output,prefix=name+'-') as t:
            v=Path(t)/'vault';shutil.copytree(vault,v);bad=copy.deepcopy(root);edit(bad)
            raw=canon(bad);pin=sha(raw);(v/'snapshots'/pin).write_bytes(raw)
            rejects(name,lambda:roots(v,pin,'GLYPH_BOUNDED_IMPORT_STREAM_V2',100000000),ValueError,lambda e:str(e)==reason)
    # A producer-pinned chain with inflated ancestor bytes still needs actual
    # native restoration and independent source identity, not a metadata sum.
    def inflated(v):
        r=copy.deepcopy(root);prev=parse(read(v/'snapshots'/parent,65536))
        d=next(d for d in prev['documents'] if d['sha256']==old['sha256']);d['bytes']+=1
        raw=canon(prev);pin=sha(raw);(v/'snapshots'/pin).write_bytes(raw)
        r['parent']=pin;raw=canon(r);new=sha(raw);(v/'snapshots'/new).write_bytes(raw);(v/'CURRENT').write_text(new+'\n')
    pool('metadata-only-ancestor-size',inflated,ValueError,lambda e:str(e)=='historical source binding',lambda v:authenticate(v,(v/'CURRENT').read_text().strip()))
    def metadata(v,mode):
        obj=v/'objects'/old['sha256'];r=parse(read(obj/'root.json',8192));s=json.loads((obj/'storage.json').read_text())
        if mode=='codec':s['ef']='invented-codec'
        elif mode=='size':r['artifacts']['ef']['bytes']=1<<40
        elif mode=='bool':r['artifacts']['ef']['bytes']=True
        else:
            with tempfile.TemporaryDirectory() as t:
                expanded=Path(t)/'object';expand(obj,expanded);payload=(expanded/'ef').read_bytes()
            (obj/'ef.packed').write_bytes(zlib.compress(payload,9));s['ef']='deflate9';r['artifacts']['ef']['bytes']=1
        (obj/'storage.json').write_bytes(canon(s));(obj/'root.json').write_bytes(canon(r))
    for mode,reason in [('codec','unknown codec'),('size','component budget'),('bool','component budget'),('cap','incomplete codec stream')]:
        def operation(v):
            with tempfile.TemporaryDirectory() as t:expand(v/'objects'/old['sha256'],Path(t)/'object')
        pool('packed-'+mode,lambda v:metadata(v,mode),ValueError,lambda e:str(e)==reason,operation)
    total=sum(p.stat().st_size for p in vault.rglob('*') if p.is_file());budget=sum(d['bytes'] for d in versions.values())
    for name,path in [('required-history-sidecar-budget','required-history-sidecar'),('unreachable-object-budget','objects/'+('f'*64)+'/unreachable.packed')]:
        def exhaustion(v):
            p=v/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'\0'*(budget-total+1))
        pool(name,exhaustion,ValueError,lambda e:str(e)=='complete authenticated history budget exceeded')
    (output/'result.json').write_text(json.dumps(records,sort_keys=True,indent=2)+'\n')
    return records
