"""Cooperative local SIGKILL recovery, only explicitly journal-owned additions.

No GC, power-loss guarantee or deletion of unknown staging/retained history.
All calls require the existing Vault LOCK. Published transactions keep objects.
"""
import json,os,shutil
from vault import canon,sha,syncdir,write

FORMAT='GLYPH_PACKED_TRANSACTION_V0'

def save(stage,data):
    temporary=stage/'transaction.next'
    write(temporary,canon(data));os.replace(temporary,stage/'transaction.json');syncdir(stage)

def start(stage,parent):
    save(stage,dict(format=FORMAT,parent=parent,objects=[],snapshot=None,target=None))

def track(stage,kind,value):
    data=json.loads((stage/'transaction.json').read_bytes())
    if kind=='objects': data[kind].append(value)
    else: data[kind]=value
    save(stage,data)

def recover(vault,stage):
    journal=stage/'transaction.json'
    if not journal.exists(): return False # unknown/legacy staging is retained
    def pin(x): return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
    if stage.is_symlink() or journal.is_symlink() or not journal.is_file() or journal.stat().st_size>16384:
        raise ValueError('transaction journal admission')
    raw=journal.read_bytes();data=json.loads(raw)
    fields={'format','parent','objects','snapshot'}
    if canon(data)!=raw or set(data) not in (fields,fields|{'target'}) or data['format']!=FORMAT:
        raise ValueError('transaction journal schema')
    target=data.get('target',data['snapshot']) # read earlier owned journals
    if not (data['parent'] is None or pin(data['parent'])) or not (data['snapshot'] is None or pin(data['snapshot'])) or not (target is None or pin(target)):
        raise ValueError('transaction journal pin')
    if type(data['objects']) is not list or len(data['objects'])>64 or len(set(data['objects']))!=len(data['objects']) or not all(pin(x) for x in data['objects']):
        raise ValueError('transaction owned object list')
    current=vault.current()
    if current is not None and current==target:
        vault.snapshot(current) # published: never remove its objects or root
    elif current==data['parent']:
        # Independently guard every committed ancestor against journal mistakes.
        protected_objects=set();protected_roots=set();cur=current
        while cur:
            root=vault.snapshot(cur);protected_roots.add(cur)
            protected_objects.update(d['sha256'] for d in root['documents']);cur=root['parent']
        if protected_objects.intersection(data['objects']) or data['snapshot'] in protected_roots:
            raise ValueError('transaction attempted committed history removal')
        if data['snapshot']:
            path=vault.path/'snapshots'/data['snapshot']
            if path.exists():
                if path.is_symlink() or not path.is_file() or sha(path.read_bytes())!=data['snapshot']:
                    raise ValueError('transaction snapshot identity')
                path.unlink();syncdir(path.parent)
        for key in data['objects']:
            path=vault.path/'objects'/key
            if path.exists():
                if path.is_symlink() or not path.is_dir():raise ValueError('transaction object directory')
                shutil.rmtree(path)
        syncdir(vault.path/'objects')
    else: raise ValueError('transaction CURRENT conflict')
    return True
