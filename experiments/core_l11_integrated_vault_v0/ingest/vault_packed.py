"""Opt-in packed storage over existing STREAM_V2 logical contract. Research only."""
import json,shutil,sys,tempfile
from pathlib import Path
from vault import Vault,AcceptanceRejected,canon
from vault_scale import VaultScale
from packed_storage import expand,view
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'experiments/personal_memory_1tb_v2'))
from verified_hybrid_archive import encode_best

class VaultPacked(VaultScale):
    def build_physical(self,data,leaf,decoder):
        super().build_physical(data,leaf,decoder)
        codecs={}
        for name in ('rlb2','rlr2','ef'):
            src=leaf/name;codec,payload=encode_best(src.read_bytes())
            (leaf/(name+'.packed')).write_bytes(payload);src.unlink();codecs[name]=codec
        (leaf/'storage.json').write_bytes(canon(codecs))
    def restore_object(self,obj,key,size,out):
        with tempfile.TemporaryDirectory() as t:
            expanded=Path(t)/'object';expand(obj,expanded)
            super().restore_object(expanded,key,size,out)
    def validate_object(self,d):
        if not d['bytes']: return super().validate_object(d)
        with tempfile.TemporaryDirectory() as t:
            v=Path(t);(v/'objects').mkdir();expand(self.path/'objects'/d['sha256'],v/'objects'/d['sha256'])
            VaultScale(v,self.ref).validate_object(d)
    def query(self,pin,pattern,limit=256):
        with tempfile.TemporaryDirectory() as t:
            v=Path(t);view(self.path,v)
            return VaultScale(v,self.ref).query(pin,pattern,limit)
    def accept_candidate(self,root,pin):
        # Budget counts every unique retained source version once, not duplicate
        # logical slots. All retained files incl old snapshots/objects count.
        from snapshot_contract import load_chain
        versions={};cur=pin
        while cur:
            r=load_chain(self.path,cur,self.FORMAT,self.SOURCE_LIMIT)
            for d in r['documents']: versions[d['sha256']]=d['bytes']
            cur=r['parent']
        budget=sum(versions.values());total=65 # projected CURRENT
        for p in self.path.rglob('*'):
            if p.relative_to(self.path).parts[0]=='staging' or p==self.path/'CURRENT': continue
            if p.is_symlink(): raise AcceptanceRejected('symlink in retained candidate')
            if p.is_file(): total+=p.stat().st_size
        if total>budget: raise AcceptanceRejected('full retained storage budget exceeded: %d > %d'%(total,budget))
