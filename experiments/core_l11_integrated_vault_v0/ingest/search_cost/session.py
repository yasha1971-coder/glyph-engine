"""Opt-in research session. Existing storage/receipts/default query unchanged.

Reuse is conditional on full live/private/code SHA guards. One explicit root,
quiescent Vault, no retarget, daemon, persisted cache or freshness assertion.
"""
import hashlib,os,stat,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;INGEST=HERE.parent;REPO=HERE.parents[3]
sys.path.insert(0,str(INGEST))
from vault import canon
from vault_scale import VaultScale
from scale_leaf import QueryState
from packed_storage import view


def require(ok,message):
    if not ok:raise ValueError(message)

def catalog(root):
    require(stat.S_ISDIR(root.lstat().st_mode),'session directory admission')
    out={}
    for p in sorted(root.rglob('*')):
        s=p.lstat()
        if stat.S_ISDIR(s.st_mode):continue
        require(stat.S_ISREG(s.st_mode),'nonregular session entry')
        require(s.st_size<=128<<20,'session file admission')
        h=hashlib.sha256();size=0
        with p.open('rb') as f:
            for raw in iter(lambda:f.read(1<<20),b''):h.update(raw);size+=len(raw)
        require(size==s.st_size,'session input changed during read')
        out[str(p.relative_to(root))]=dict(bytes=size,sha256=h.hexdigest())
    return out

def code_catalog(reference):
    paths=[*INGEST.glob('*.py'),*HERE.glob('*.py'),
        *(REPO/'experiments/loc_text_ef_v0'/n for n in ('trusted_materialize_v0.py','query_ef_research.py','self_index_exact_range_extractor.py')),
        REPO/'experiments/core_l9_exact_query_v0/producer.py',REPO/'experiments/core_l9_scale_v0/verifier.py',
        *(reference/'tools'/n for n in ('rlbwt_query_v2.py','rlbwt_container_v2.py','rlbwt_rank_v2.py'))]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

class Session:
    def __init__(self,vault,pin,reference,temp_budget=32<<20,object_limit=4,source_budget=8<<20):
        self.vault,self.pin,self.reference=Path(vault),pin,Path(reference)
        self.states={};self.stage=None;self.closed=True;started=time.monotonic()
        self.live=catalog(self.vault);self.codes=code_catalog(self.reference)
        self.root=VaultScale(self.vault,self.reference).snapshot(pin)
        docs=self.root['documents'];unique={d['sha256']:d for d in docs if d['bytes']}
        require(len(unique)<=object_limit,'active object admission')
        require(sum(d['bytes'] for d in unique.values())<=source_budget,'active source admission')
        # Actual decoded lengths from all retained physical roots; native state
        # snapshots and nested source-auth copies are paid, not free cache data.
        import json
        expanded=0;active=[]
        for obj in (self.vault/'objects').iterdir():
            if (obj/'empty.json').exists():continue
            r=json.loads((obj/'root.json').read_bytes())
            amount=sum(x['bytes'] for x in r['artifacts'].values())
            require(type(amount) is int and 0<amount<=3*(128<<20),'session component admission')
            expanded+=amount
            if obj.name in unique:active.append(amount+(self.vault/'decoder').stat().st_size)
        metadata=sum(x['bytes'] for n,x in self.live.items() if not n.endswith('.packed'))
        estimate=expanded+metadata+sum(active)+(max(active) if active else 0)+(max((d['bytes'] for d in unique.values()),default=0))
        require(estimate<=temp_budget,'session temporary byte admission')
        self.metrics=dict(unique_active_objects=len(unique),logical_nonempty_documents=sum(d['bytes']>0 for d in docs),
            admitted_temporary_file_bytes=estimate,native_source_bytes_once=sum(d['bytes'] for d in unique.values()))
        try:
            self.stage=tempfile.TemporaryDirectory(prefix='packed-query-session-');self.expanded=Path(self.stage.name)
            at=time.monotonic();view(self.vault,self.expanded);self.metrics['expand_seconds']=time.monotonic()-at
            v=VaultScale(self.expanded,self.reference)
            # Match original full root/decoder/empty/physical checks, not just
            # file-existence flags. Root format and magic are unchanged.
            from vault import digestfile
            require(digestfile(self.expanded/'decoder')==self.root['decoder_sha256'],'session decoder identity')
            at=time.monotonic()
            for d in docs:
                v.validate_object(d)
                if d['bytes'] and d['sha256'] not in self.states:
                    obj=self.expanded/'objects'/d['sha256']
                    self.states[d['sha256']]=QueryState(obj/'root.json',d['manifest'],obj/'rlb2',obj/'rlr2',obj/'ef',self.expanded/'decoder',self.reference)
            self.metrics['state_open_seconds']=time.monotonic()-at
            self.private=self._private();self.closed=False
            require(catalog(self.vault)==self.live,'live Vault changed during open')
            require(code_catalog(self.reference)==self.codes,'code changed during open')
            self.metrics['open_seconds']=time.monotonic()-started
            self.metrics['leaf_startups']={k:s.startup for k,s in self.states.items()}
        except BaseException:self.close();raise
    def _private(self):
        result={'expanded':catalog(self.expanded)}
        for key,s in self.states.items():result[key]=catalog(Path(s.stage.name))
        return result
    def _guard(self):
        require(subprocess.check_output(['git','-C',str(self.reference),'rev-parse','HEAD'],text=True).strip()=='c33cb493bc75f13ee4a904fabb190a29a196c94b',
                'frozen reference HEAD changed; reopen session')
        require(catalog(self.vault)==self.live,'live Vault changed; reopen session')
        require(self._private()==self.private,'private snapshot changed; reopen session')
        require(code_catalog(self.reference)==self.codes,'code changed; reopen session')
    def query(self,expected_pin,pattern,limit=256):
        require(not self.closed,'session closed')
        require(expected_pin==self.pin,'caller snapshot pin mismatch')
        require(type(pattern) is bytes and 0<len(pattern)<=4097 and type(limit) is int and 0<=limit<=256,'query policy')
        started=time.monotonic();self._guard();guard=time.monotonic()-started
        count=0;coords=[];high=False;leaf_metrics=[]
        # Logical duplicates remain distinct; physical QueryState sharing never
        # collapses document slots/count. Query each doc just as original does.
        for d in self.root['documents']:
            if not d['bytes']:continue
            state=self.states[d['sha256']];r=state.query(pattern,256)
            count+=r['count'];high|=r['count']>256
            coords.extend([d['doc_id'],d['revision'],off] for off in r['offsets'])
            leaf_metrics.append(state.last_metrics)
        at=time.monotonic();self._guard();guard+=time.monotonic()-at
        claim=dict(format='GLYPH_STREAM_EXACT_RECEIPT_V2',snapshot=self.pin,query_hex=pattern.hex(),limit=limit,count=count,
            coordinates=[] if high else coords[:limit],enumeration='OMITTED_HIGH_COUNT' if high else 'PREFIX' if count>limit else 'COMPLETE',
            status='UNTRUSTED' if high or count>limit else 'FOUND' if count else 'PROVEN_EMPTY',
            covered_doc_ids=[d['doc_id'] for d in self.root['documents']],independent_index_source_semantics=False)
        self.last_metrics=dict(wall_seconds=time.monotonic()-started,full_guards_seconds=guard,leaf_metrics=leaf_metrics)
        return claim
    def close(self):
        for s in self.states.values():s.close()
        self.states={}
        if self.stage is not None:self.stage.cleanup();self.stage=None
        self.closed=True
