"""Explicit 4KiB research profile. V0 admission remains unchanged."""
from vault import Vault, QueryState, digestfile
class Vault4K(Vault):
    SOURCE_LIMIT=4096
    FORMAT='GLYPH_BOUNDED_IMPORT_4K_V1'
    def query(self,pin,pattern,limit=256):
        if type(pattern) is not bytes or not 0<len(pattern)<=4097 or type(limit) is not int or not 0<=limit<=256: raise ValueError('query policy')
        root=self.snapshot(pin)
        if digestfile(self.path/'decoder')!=root['decoder_sha256']: raise ValueError('decoder mismatch')
        coordinates=[]; count=0; covered=[]
        for doc in root['documents']:
            self.validate_object(doc); covered.append(doc['doc_id'])
            if not doc['bytes']: continue
            obj=self.path/'objects'/doc['sha256']
            state=QueryState(obj/'root.json',doc['manifest'],obj/'rlb2',obj/'rlr2',obj/'ef',self.path/'decoder',self.ref)
            try:
                # Complete bounded internal enumeration, <=4096 offsets/leaf.
                result=state.rt.query(pattern,-1)
                offsets=result['locate_offsets']; n=result['count']
                if offsets!=sorted(set(offsets)) or len(offsets)!=n or n>4096: raise ValueError('incomplete locate')
                for off in offsets:
                    if not 0<=off<=doc['bytes']-len(pattern): raise ValueError('coordinate bound')
                count+=n
                remaining=limit-len(coordinates)
                coordinates.extend([doc['doc_id'],doc['revision'],off] for off in offsets[:remaining])
            finally: state.close()
        return dict(format='GLYPH_4K_EXACT_RECEIPT_V1',snapshot=pin,vault_id=root['vault_id'],query_hex=pattern.hex(),limit=limit,count=count,coordinates=coordinates,covered_doc_ids=covered,coverage_complete=True,enumeration_complete=count<=limit,status='UNTRUSTED' if count>limit else 'FOUND' if count else 'PROVEN_EMPTY')
