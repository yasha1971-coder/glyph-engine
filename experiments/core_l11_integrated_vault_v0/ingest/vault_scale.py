"""Opt-in streaming verification profile; no compact index-source proof claim."""
from pathlib import Path
from vault import Vault,digestfile
from physical import build_object
from scale_leaf import QueryState
class VaultScale(Vault):
    SOURCE_LIMIT=100000000
    COMPONENT_LIMIT=128<<20
    FORMAT='GLYPH_BOUNDED_IMPORT_STREAM_V2'
    def build_physical(self,data,leaf,decoder):build_object(data,leaf,self.ref,decoder,self.SOURCE_LIMIT)
    def query(self,pin,pattern,limit=256):
        if type(pattern) is not bytes or not 0<len(pattern)<=4097 or type(limit) is not int or not 0<=limit<=256:raise ValueError('query policy')
        root=self.snapshot(pin);count=0;coords=[];high=False
        if digestfile(self.path/'decoder')!=root['decoder_sha256']:raise ValueError('decoder changed')
        for d in root['documents']:
            self.validate_object(d)
            if not d['bytes']:continue
            obj=self.path/'objects'/d['sha256']
            state=QueryState(obj/'root.json',d['manifest'],obj/'rlb2',obj/'rlr2',obj/'ef',self.path/'decoder',self.ref)
            try:
                r=state.query(pattern,256);count+=r['count'];high|=r['count']>256
                coords.extend([d['doc_id'],d['revision'],off] for off in r['offsets'])
            finally:state.close()
        return dict(format='GLYPH_STREAM_EXACT_RECEIPT_V2',snapshot=pin,query_hex=pattern.hex(),limit=limit,count=count,coordinates=[] if high else coords[:limit],enumeration='OMITTED_HIGH_COUNT' if high else 'PREFIX' if count>limit else 'COMPLETE',status='UNTRUSTED' if high or count>limit else 'FOUND' if count else 'PROVEN_EMPTY',covered_doc_ids=[d['doc_id'] for d in root['documents']],independent_index_source_semantics=False)
