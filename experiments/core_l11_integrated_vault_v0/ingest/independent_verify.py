"""Stdlib-only logical verifier. Restores sources through L8, replays L9 proofs.
No imports from importer, builder, snapshot_contract or query producer.
Caller supplies expected root/query/policy, independently of the claim.
"""
import argparse, hashlib, json, stat, subprocess, sys, tempfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]

def demand(ok,why):
    if not ok: raise ValueError(why)
def sha(b): return hashlib.sha256(b).hexdigest()
def canon(x): return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pairs(xs):
    out={}
    for k,v in xs: demand(k not in out,'duplicate field'); out[k]=v
    return out
def read(path,cap):
    demand(not path.parent.is_symlink() and stat.S_ISREG(path.lstat().st_mode),'regular file required')
    with path.open('rb') as f: raw=f.read(cap+1)
    demand(len(raw)<=cap,'bounded read'); return raw
def parse(raw):
    d=json.loads(raw,object_pairs_hook=pairs); demand(canon(d)==raw,'canonical JSON required'); return d
def ishex(s,n): return type(s) is str and len(s)==n and set(s)<=set('0123456789abcdef')
def integer(x,n): return type(x) is int and 0<=x<=n

def roots(vault,pin):
    arr=[]; seen=set(); p=pin
    while p is not None:
        demand(ishex(p,64) and p not in seen and len(arr)<64,'chain bound/pin')
        seen.add(p); raw=read(vault/'snapshots'/p,65536); demand(sha(raw)==p,'wrong snapshot')
        r=parse(raw)
        demand(type(r) is dict and set(r)=={'format','vault_id','parent','documents','decoder_sha256'},'root fields')
        demand(r['format']=='GLYPH_BOUNDED_IMPORT_V0' and ishex(r['vault_id'],32) and ishex(r['decoder_sha256'],64),'root identity')
        demand(type(r['documents']) is list and len(r['documents'])<=64,'document count')
        names=[]
        for slot,d in enumerate(r['documents']):
            demand(type(d) is dict and set(d)=={'doc_id','revision','path_hex','sha256','bytes','manifest'},'document fields')
            demand(integer(d['doc_id'],63) and d['doc_id']==slot and integer(d['revision'],2**64-1) and integer(d['bytes'],256) and ishex(d['sha256'],64),'document types')
            h=d['path_hex']; demand(type(h) is str and 0<len(h)<=510 and len(h)%2==0,'path bound')
            b=bytes.fromhex(h); demand(b.hex()==h and b not in (b'.',b'..') and b'/' not in b and b'\0' not in b and b not in names,'path identity'); names.append(b)
            demand(ishex(d['manifest'],64) if d['bytes'] else d['manifest'] is None and d['sha256']==sha(b''),'source binding')
        arr.append(r); p=r['parent']
    chronological=arr[::-1]
    demand(all(d['revision']==0 for d in chronological[0]['documents']),'initial revision')
    for prev,r in zip(chronological,chronological[1:]):
        demand(r['vault_id']==prev['vault_id'] and r['decoder_sha256']==prev['decoder_sha256'] and len(r['documents'])>=len(prev['documents']),'lineage identity')
        for i,d in enumerate(r['documents']):
            if i>=len(prev['documents']): demand(d['revision']==0,'new revision')
            else: demand(d==prev['documents'][i] or d['revision']==prev['documents'][i]['revision']+1,'revision progression')
    return arr[0]

def verify(vault,pin,pattern,limit,claim,reference,selected=None,output=None):
    demand(type(pattern) is bytes and 0<len(pattern)<=257 and integer(limit,256),'query policy')
    r=roots(vault,pin)
    demand(set(claim)=={'snapshot','query_hex','limit','count','coordinates','status','physical_receipts'},'claim fields')
    demand(claim['snapshot']==pin and claim['query_hex']==pattern.hex() and type(claim['limit']) is int and claim['limit']==limit,'query/snapshot binding')
    decoder=vault/'decoder'; demand(sha(read(decoder,1<<20))==r['decoder_sha256'],'decoder pin')
    demand(type(claim['physical_receipts']) is list and len(claim['physical_receipts'])==sum(d['bytes']>0 for d in r['documents']),'receipt coverage')
    coords=[]; replay=0; selected_bytes=None
    with tempfile.TemporaryDirectory(prefix='glyph-independent-') as tmp:
        tmp=Path(tmp)
        for d in r['documents']:
            obj=vault/'objects'/d['sha256']; demand(obj.is_dir() and not obj.is_symlink(),'missing/linked leaf')
            raw=b''
            if d['bytes']:
                manifest=read(obj/'root.json',8192); demand(sha(manifest)==d['manifest'],'manifest pin')
                physical=parse(manifest); demand(physical['source']=={'bytes':d['bytes'],'sha256':d['sha256']},'logical source binding')
                restored=tmp/(str(d['doc_id'])+'.raw')
                args=[sys.executable,str(REPO/'experiments/loc_text_ef_v0/trusted_materialize_v0.py'),'--manifest',str(obj/'root.json'),'--root-sha256',d['manifest'],'--rlb2',str(obj/'rlb2'),'--rlr2',str(obj/'rlr2'),'--ef',str(obj/'ef'),'--decoder',str(decoder),'--output',str(restored),'--threads','2','--mode','rank']
                subprocess.run(args,check=True,capture_output=True,timeout=90)
                raw=read(restored,256)
                receipt=tmp/'receipt.json'; receipt.write_bytes(canon(claim['physical_receipts'][replay]))
                args=[sys.executable,str(REPO/'experiments/core_l9_exact_query_v0/verifier.py'),'--manifest',str(obj/'root.json'),'--root-sha256',d['manifest'],'--rlb2',str(obj/'rlb2'),'--rlr2',str(obj/'rlr2'),'--ef',str(obj/'ef'),'--decoder',str(decoder),'--materialized',str(restored),'--receipt',str(receipt),'--pattern-hex',pattern.hex(),'--limit','256','--research',str(REPO),'--reference',str(reference)]
                subprocess.run(args,check=True,capture_output=True,timeout=90); replay+=1
            else:
                demand(read(obj/'empty.json',512)==canon({'format':'GLYPH_EMPTY_SOURCE_V0','sha256':d['sha256'],'bytes':0}),'explicit empty source')
            demand(len(raw)==d['bytes'] and sha(raw)==d['sha256'],'restored identity')
            coords.extend([d['doc_id'],d['revision'],i] for i in range(len(raw)) if raw.startswith(pattern,i))
            if d['doc_id']==selected: selected_bytes=raw
    expected={'snapshot':pin,'query_hex':pattern.hex(),'limit':limit,'count':len(coords),'coordinates':coords[:limit],'status':'UNTRUSTED' if len(coords)>limit else 'FOUND' if coords else 'PROVEN_EMPTY'}
    demand(canon({k:claim[k] for k in expected})==canon(expected),'logical answer mismatch')
    if selected is not None:
        demand(selected_bytes is not None and output is not None,'selected document absent')
        with Path(output).open('xb') as f: f.write(selected_bytes)
    return {'status':'BOUNDED_INGEST_INDEPENDENT_PASS','documents_verified':len(r['documents']),'physical_replays':replay,'count':len(coords)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('vault','pin','query-hex','claim','reference'): p.add_argument('--'+n,required=True)
    p.add_argument('--limit',type=int,default=256); p.add_argument('--selected',type=int); p.add_argument('--output')
    a=p.parse_args()
    try: print(json.dumps(verify(Path(a.vault),a.pin,bytes.fromhex(a.query_hex),a.limit,parse(read(Path(a.claim),1<<23)),Path(a.reference),a.selected,a.output),sort_keys=True))
    except Exception as e:
        print(json.dumps({'status':'UNTRUSTED','reason':str(e)[:300]}),file=sys.stderr); sys.exit(3)
