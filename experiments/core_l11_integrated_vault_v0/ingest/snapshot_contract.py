"""Producer-side bounded schema/lineage validation; verifier has its own checks."""
import hashlib, json, stat
from pathlib import Path

def canonical(x): return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def check(ok,why):
    if not ok: raise ValueError(why)
def hexstr(x,n): return type(x) is str and len(x)==n and all(c in '0123456789abcdef' for c in x)
def number(x,n): return type(x) is int and 0<=x<=n
def unique(pairs):
    d={}
    for k,v in pairs: check(k not in d,'duplicate field'); d[k]=v
    return d

def load_chain(vault,pin):
    current=pin; chain=[]; seen=set()
    while current is not None:
        check(hexstr(current,64) and current not in seen and len(chain)<64,'snapshot chain admission')
        seen.add(current); path=Path(vault)/'snapshots'/current
        check(not path.parent.is_symlink() and stat.S_ISREG(path.lstat().st_mode),'snapshot regular file')
        with path.open('rb') as f: raw=f.read(65537)
        check(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==current,'snapshot pin/size')
        r=json.loads(raw,object_pairs_hook=unique)
        check(canonical(r)==raw and type(r) is dict and set(r)=={'format','vault_id','parent','documents','decoder_sha256'},'snapshot canonical schema')
        check(r['format']=='GLYPH_BOUNDED_IMPORT_V0' and hexstr(r['vault_id'],32) and hexstr(r['decoder_sha256'],64),'snapshot identity')
        docs=r['documents']; check(type(docs) is list and len(docs)<=64,'document bound')
        paths=set()
        for i,d in enumerate(docs):
            check(type(d) is dict and set(d)=={'doc_id','revision','path_hex','sha256','bytes','manifest'},'document schema')
            check(number(d['doc_id'],63) and d['doc_id']==i and number(d['revision'],2**64-1) and number(d['bytes'],256) and hexstr(d['sha256'],64),'document identity')
            h=d['path_hex']; check(type(h) is str and 0<len(h)<=510 and len(h)%2==0,'path admission')
            path=bytes.fromhex(h)
            check(path.hex()==h and b'/' not in path and b'\0' not in path and path not in (b'.',b'..') and h not in paths,'path safety/uniqueness'); paths.add(h)
            check((hexstr(d['manifest'],64) if d['bytes'] else d['manifest'] is None and d['sha256']==hashlib.sha256(b'').hexdigest()),'empty/nonempty binding')
        chain.append(r); current=r['parent']
    for old,new in zip(reversed(chain),list(reversed(chain))[1:]):
        check(old['vault_id']==new['vault_id'] and old['decoder_sha256']==new['decoder_sha256'],'vault lineage')
        check(len(new['documents'])>=len(old['documents']),'removed logical slot')
        for i,d in enumerate(new['documents']):
            if i>=len(old['documents']): check(d['revision']==0,'new doc revision')
            else:
                prev=old['documents'][i]
                check(d==prev or d['revision']==prev['revision']+1,'revision lineage')
    check(all(d['revision']==0 for d in chain[-1]['documents']),'initial revisions')
    return chain[0]
