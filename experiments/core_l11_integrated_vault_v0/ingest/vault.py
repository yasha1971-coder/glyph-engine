"""Nonnormative bounded immutable snapshot store. Linux/local filesystem only."""
import fcntl, hashlib, json, os, shutil, signal, stat, subprocess, sys, tempfile, uuid
from pathlib import Path
from physical import build_object
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'experiments/core_l9_exact_query_v0'))
from producer import QueryState

def sha(b): return hashlib.sha256(b).hexdigest()
def canon(x): return (json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode()
def syncdir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def write(p,b):
    with p.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
def crash(point):
    if os.environ.get('GLYPH_TEST_CRASH')==point: os.kill(os.getpid(),signal.SIGKILL)
def digestfile(p):
    if not stat.S_ISREG(p.lstat().st_mode): raise ValueError('regular immutable file required')
    if p.stat().st_size>1<<20: raise ValueError('physical component size admission')
    with p.open('rb') as f: raw=f.read((1<<20)+1)
    if len(raw)>1<<20: raise ValueError('physical component grew')
    return sha(raw)
def read_source(folder,name,cap=256):
    raw=os.fsencode(name)
    if not raw or b'/' in raw or raw in (b'.',b'..'): raise ValueError('one-level relative filename required')
    fd=os.open(folder/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or a.st_size>cap: raise ValueError('source admission')
        b=os.read(fd,cap+1); z=os.fstat(fd)
        if len(b)>cap or len(b)!=a.st_size or (a.st_size,a.st_mtime_ns,a.st_ctime_ns,a.st_ino)!=(z.st_size,z.st_mtime_ns,z.st_ctime_ns,z.st_ino): raise ValueError('source changed')
        return b
    finally: os.close(fd)

class Vault:
    SOURCE_LIMIT=256
    FORMAT="GLYPH_BOUNDED_IMPORT_V0"
    def __init__(self,path,reference): self.path=Path(path); self.ref=Path(reference)
    def current(self):
        p=self.path/'CURRENT'
        return p.read_text().strip() if p.exists() else None
    def snapshot(self,pin):
        from snapshot_contract import load_chain
        return load_chain(self.path,pin,self.FORMAT,self.SOURCE_LIMIT)
    def import_files(self,source,changes,expected,decoder):
        # changes: explicit {name, doc_id?}; absent paths are never deleted.
        p=self.path
        p.mkdir(exist_ok=True)
        for n in ('objects','snapshots','staging'): (p/n).mkdir(exist_ok=True)
        syncdir(p); syncdir(p.parent)
        with (p/'LOCK').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if self.current()!=expected: raise ValueError('writer parent conflict')
            old=self.snapshot(expected) if expected else None
            if old and digestfile(p/'decoder')!=old['decoder_sha256']: raise ValueError('decoder changed since parent')
            docs=[dict(x) for x in old['documents']] if old else []
            if not (p/'decoder').exists():
                write(p/'decoder',Path(decoder).read_bytes()); os.chmod(p/'decoder',0o700)
                with (p/'decoder').open('rb') as file: os.fsync(file.fileno())
                syncdir(p)
            staged=Path(tempfile.mkdtemp(dir=p/'staging'))
            try:
                touched=set()
                for change in changes:
                    name=change['name']; data=read_source(Path(source),name,self.SOURCE_LIMIT)
                    ident=change.get('doc_id',len(docs))
                    if type(ident) is not int or not 0<=ident<=len(docs) or ident in touched: raise ValueError('explicit stable identity required')
                    touched.add(ident)
                    key=sha(data); obj=p/'objects'/key
                    if not obj.exists():
                        leaf=staged/key
                        if data:
                            build_object(data,leaf,self.ref,p/'decoder')
                            self.restore_object(leaf,key,len(data),staged/('check-'+key))
                        else:
                            leaf.mkdir(); write(leaf/'empty.json',canon({'format':'GLYPH_EMPTY_SOURCE_V0','sha256':key,'bytes':0}))
                        for file in leaf.iterdir():
                            with file.open('rb') as f: os.fsync(f.fileno())
                        syncdir(leaf); crash('object-ready')
                        os.rename(leaf,obj); syncdir(p/'objects')
                    manifest=sha((obj/'root.json').read_bytes()) if data else None
                    doc=dict(doc_id=ident,revision=docs[ident]['revision']+1 if ident<len(docs) else 0,path_hex=os.fsencode(name).hex(),sha256=key,bytes=len(data),manifest=manifest)
                    self.validate_object(doc)
                    if ident==len(docs): docs.append(doc)
                    else: docs[ident]=doc
                if len(docs)>64 or len({d['path_hex'] for d in docs})!=len(docs): raise ValueError('document/path bound')
                for document in docs: self.validate_object(document)
                root=dict(format=self.FORMAT,vault_id=old['vault_id'] if old else uuid.uuid4().hex,parent=expected,documents=docs,decoder_sha256=digestfile(p/'decoder'))
                raw=canon(root); pin=sha(raw)
                snapshot=p/'snapshots'/pin
                if snapshot.exists():
                    if snapshot.read_bytes()!=raw: raise ValueError('immutable snapshot conflict')
                    with snapshot.open('rb') as f: os.fsync(f.fileno())
                else: write(snapshot,raw)
                self.snapshot(pin)
                syncdir(p/'snapshots'); crash('snapshot-ready')
                head=staged/'CURRENT'; write(head,(pin+'\n').encode()); crash('before-current')
                os.replace(head,p/'CURRENT'); crash('after-current'); syncdir(p); crash('published')
                return pin
            finally: shutil.rmtree(staged)
    def validate_object(self,d):
        obj=self.path/'objects'/d['sha256']
        if obj.is_symlink() or not obj.is_dir(): raise ValueError('invalid object directory')
        if d['bytes']==0:
            if d['sha256']!=sha(b'') or d['manifest'] is not None or not obj.is_dir(): raise ValueError('empty identity')
            if (obj/'empty.json').is_symlink() or (obj/'empty.json').stat().st_size>512 or (obj/'empty.json').read_bytes()!=canon({'format':'GLYPH_EMPTY_SOURCE_V0','sha256':d['sha256'],'bytes':0}): raise ValueError('explicit empty binding')
            return
        if not stat.S_ISREG((obj/'root.json').lstat().st_mode) or (obj/'root.json').stat().st_size>8192: raise ValueError('invalid manifest file')
        raw=(obj/'root.json').read_bytes()
        if sha(raw)!=d['manifest']: raise ValueError('physical root mismatch')
        root=json.loads(raw)
        if root['source']!={'bytes':d['bytes'],'sha256':d['sha256']}: raise ValueError('source binding')
        if set(root['artifacts'])!={'rlb2','rlr2','ef'}: raise ValueError('mandatory physical artifacts absent')
        for n,rec in root['artifacts'].items():
            if n not in ('rlb2','rlr2','ef') or digestfile(obj/n)!=rec['sha256'] or (obj/n).stat().st_size!=rec['bytes']: raise ValueError('incomplete physical coverage')
    def query(self,pin,pattern,limit=256):
        if type(pattern) is not bytes or not 0<len(pattern)<=257 or type(limit) is not int or not 0<=limit<=256: raise ValueError('query admission')
        root=self.snapshot(pin); coords=[]; receipts=[]
        if digestfile(self.path/'decoder')!=root['decoder_sha256']: raise ValueError('decoder mismatch')
        for d in root['documents']:
            self.validate_object(d)
            if not d['bytes']: continue
            obj=self.path/'objects'/d['sha256']
            s=QueryState(obj/'root.json',d['manifest'],obj/'rlb2',obj/'rlr2',obj/'ef',self.path/'decoder',self.ref)
            try:
                r=s.query(pattern,256)
                if not r['coverage_complete']: raise ValueError('incomplete physical enumeration')
                coords.extend([d['doc_id'],d['revision'],off] for off in r['offsets']); receipts.append(r)
            finally: s.close()
        return dict(snapshot=pin,query_hex=pattern.hex(),limit=limit,count=len(coords),coordinates=coords[:limit],status='UNTRUSTED' if len(coords)>limit else 'FOUND' if coords else 'PROVEN_EMPTY',physical_receipts=receipts)
    def restore_object(self,obj,key,size,out):
        args=[sys.executable,str(REPO/'experiments/loc_text_ef_v0/trusted_materialize_v0.py'),'--manifest',str(obj/'root.json'),'--root-sha256',digestfile(obj/'root.json'),'--rlb2',str(obj/'rlb2'),'--rlr2',str(obj/'rlr2'),'--ef',str(obj/'ef'),'--decoder',str(self.path/'decoder'),'--output',str(out),'--threads','2','--mode','rank']
        subprocess.run(args,check=True,capture_output=True)
        if digestfile(out)!=key or out.stat().st_size!=size: raise ValueError('independent restore mismatch')
