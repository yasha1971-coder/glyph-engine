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
    def recover(self):
        """Explicit restart under the same cooperative writer lock."""
        import fcntl
        with (self.path/'LOCK').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);self.recover_pending()
    def recover_pending(self):
        import packed_transaction as transaction
        from vault import syncdir
        for stage in (self.path/'staging').iterdir():
            if stage.is_dir() and transaction.recover(self,stage):
                shutil.rmtree(stage);syncdir(stage.parent)
    def start_transaction(self,staged,parent):
        import packed_transaction as transaction
        self._active_stage=staged
        if parent: self.validate_history(parent)
        transaction.start(staged,parent)
    def track_transaction(self,staged,kind,value):
        import packed_transaction as transaction
        transaction.track(staged,kind,value)
    def finish_transaction(self,staged):
        import packed_transaction as transaction
        transaction.recover(self,staged)
        self._active_stage=None
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
    def validate_history(self,pin):
        from snapshot_contract import load_chain
        versions={};cur=pin
        while cur:
            r=load_chain(self.path,cur,self.FORMAT,self.SOURCE_LIMIT)
            for d in r['documents']: versions[d['sha256']]=d
            cur=r['parent']
        # A version used only by an ancestor is still required preservation
        # state, not free budget metadata. Independent replay also restores it.
        for d in versions.values(): self.validate_object(d)
        return versions
    def accept_candidate(self,root,pin):
        # Budget counts every unique retained source version once, not duplicate
        # logical slots. All retained files incl old snapshots/objects count.
        versions=self.validate_history(pin)
        budget=sum(d['bytes'] for d in versions.values());total=65 # projected CURRENT
        for p in self.path.rglob('*'):
            # Only this live writer's temporary build is excluded. Unknown
            # persistent staging/orphans remain counted, never garbage-collected.
            if self._active_stage in p.parents or p==self._active_stage or p==self.path/'CURRENT': continue
            if p.is_symlink(): raise AcceptanceRejected('symlink in retained candidate')
            if p.is_file(): total+=p.stat().st_size
            elif not p.is_dir(): raise AcceptanceRejected('nonregular retained candidate')
        if total>budget: raise AcceptanceRejected('full retained storage budget exceeded: %d > %d'%(total,budget))
