"""Bounded transport decoding independent of Vault and compression producer."""
import bz2,hashlib,json,lzma,stat,zlib
from pathlib import Path
LIMIT=128<<20
def expand(obj,out):
    out.mkdir()
    for n in ('root.json','storage.json'):
        p=obj/n
        if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size>8192: raise ValueError('metadata admission')
    raw=(obj/'root.json').read_bytes()
    if len(raw)>8192: raise ValueError('root budget')
    root=json.loads(raw); codecs=json.loads((obj/'storage.json').read_text())
    if set(codecs)!={'rlb2','rlr2','ef'}: raise ValueError('storage coverage')
    (out/'root.json').write_bytes(raw)
    for name in ('rlb2','rlr2','ef'):
        p=obj/(name+'.packed'); rec=root['artifacts'][name]
        if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size>LIMIT or not 0<=rec['bytes']<=LIMIT: raise ValueError('component budget')
        data=p.read_bytes(); codec=codecs[name]
        if codec=='raw': decoded=data
        elif codec in ('xz-9','bzip2-9'):
            decoder=lzma.LZMADecompressor(memlimit=256<<20) if codec=='xz-9' else bz2.BZ2Decompressor()
            decoded=decoder.decompress(data,max_length=rec['bytes']+1)
            if not decoder.eof or decoder.unused_data: raise ValueError('incomplete codec stream')
        elif codec=='deflate9':
            decoder=zlib.decompressobj();decoded=decoder.decompress(data,rec['bytes']+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail: raise ValueError('incomplete codec stream')
        else: raise ValueError('unknown codec')
        if len(decoded)!=rec['bytes'] or hashlib.sha256(decoded).hexdigest()!=rec['sha256']: raise ValueError('decoded component identity')
        (out/name).write_bytes(decoded)

def view(vault,out):
    import shutil
    shutil.copytree(vault/'snapshots',out/'snapshots')
    shutil.copy2(vault/'decoder',out/'decoder')
    (out/'objects').mkdir()
    for obj in (vault/'objects').iterdir():
        target=out/'objects'/obj.name
        if (obj/'empty.json').exists(): shutil.copytree(obj,target)
        else: expand(obj,target)
