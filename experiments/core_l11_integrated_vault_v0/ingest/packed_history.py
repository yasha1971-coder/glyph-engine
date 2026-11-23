"""Independent complete-parent-chain budget authentication, no Vault/importer.

Every distinct committed source version is restored and hashed, including
versions absent from every query snapshot. Index assurance stays structural.
"""
import hashlib,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from independent_verify import roots,read,parse,sha,canon,demand,REPO
from independent_scale import stream
from packed_storage import expand
from storage_budget import assess

def history(vault,pin):
    demand(read(vault/'CURRENT',65)==(pin+'\n').encode(),'externally expected CURRENT')
    root=roots(vault,pin,'GLYPH_BOUNDED_IMPORT_STREAM_V2',100000000)
    demand(sha(read(vault/'decoder',1<<20))==root['decoder_sha256'],'history decoder identity')
    chain=[];versions={};cur=pin
    while cur:
        r=parse(read(vault/'snapshots'/cur,65536));chain.append(cur)
        for d in r['documents']:
            prior=versions.get(d['sha256'])
            demand(prior is None or (prior['bytes'],prior['manifest'])==(d['bytes'],d['manifest']),'conflicting source version identity')
            versions[d['sha256']]=d
        cur=r['parent']
    return root,chain,versions

def restore(vault,doc,temporary):
    obj=vault/'objects'/doc['sha256']
    demand(obj.is_dir() and not obj.is_symlink(),'missing historical source object')
    dest=temporary/'source'
    if not doc['bytes']:
        demand(read(obj/'empty.json',512)==canon(dict(format='GLYPH_EMPTY_SOURCE_V0',sha256=sha(b''),bytes=0)),'historical empty binding')
        dest.write_bytes(b'')
    else:
        expanded=temporary/'expanded';expand(obj,expanded)
        raw=read(expanded/'root.json',8192)
        demand(sha(raw)==doc['manifest'],'historical manifest pin')
        root=parse(raw)
        demand(root['source']==dict(bytes=doc['bytes'],sha256=doc['sha256']),'historical source binding')
        demand(set(root['artifacts'])=={'rlb2','rlr2','ef'},'historical index coverage')
        w=stream.Work()
        payload={n:w.file(expanded/n,root['artifacts'][n],128<<20,n!='ef') for n in ('rlb2','rlr2','ef')}
        stream.index_structure(SimpleNamespace(**{n:expanded/n for n in ('rlb2','rlr2','ef')}),root,payload,w)
        args=[sys.executable,str(REPO/'experiments/loc_text_ef_v0/trusted_materialize_v0.py'),
            '--manifest',str(expanded/'root.json'),'--root-sha256',doc['manifest'],
            '--decoder',str(vault/'decoder'),'--output',str(dest),'--threads','2','--mode','rank']
        for n in ('rlb2','rlr2','ef'):args+=['--'+n,str(expanded/n)]
        result=subprocess.run(args,capture_output=True,text=True,timeout=600)
        demand(result.returncode==0,'historical source restore: '+result.stderr[-200:])
    h=hashlib.sha256();size=0
    with dest.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):
            size+=len(data);demand(size<=doc['bytes'],'historical restored size cap');h.update(data)
    demand(size==doc['bytes'] and h.hexdigest()==doc['sha256'],'historical restored SHA/length')
    return dest

def authenticate(vault,pin):
    root,chain,versions=history(vault,pin)
    restored=[];expansion_max=0
    # assess rejects every symlink/special entry and counts ALL retained state,
    # but it cannot authenticate the denominator; do that here before PASS.
    budget=assess(vault,sum(d['bytes'] for d in versions.values()))
    for key,doc in sorted(versions.items()):
        with tempfile.TemporaryDirectory(prefix='glyph-history-') as tmp:
            tmp=Path(tmp);restore(vault,doc,tmp)
            expansion_max=max(expansion_max,sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file()))
        restored.append(dict(sha256=key,bytes=doc['bytes'],manifest=doc['manifest']))
    demand(budget['storage_budget_pass'],'complete authenticated history budget exceeded')
    return dict(status='PACKED_COMPLETE_HISTORY_PASS',head=pin,chain=chain,snapshots=len(chain),
        current_logical_bytes=sum(d['bytes'] for d in root['documents']),
        distinct_source_bytes=budget['source_bytes'],retained_bytes=budget['retained_bytes'],
        source_versions=restored,distinct_versions_restored=len(restored),
        retained_object_directories=len(list((vault/'objects').iterdir())),
        one_version_expanded_and_source_file_bytes_max=expansion_max,
        independent_index_source_semantics=False)
