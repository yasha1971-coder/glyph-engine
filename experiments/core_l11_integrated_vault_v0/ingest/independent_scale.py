"""Independent streaming full-source verification. Structural index checks only.
Does not claim independent replay of every rank prefix/EF suffix semantic relation.
"""
import argparse,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from independent_verify import roots,read,parse,sha,canon,demand,integer,REPO
spec=importlib.util.spec_from_file_location('preserved_independent_stream',REPO/'experiments/core_l9_scale_v0/verifier.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)

def verify(vault,pin,pattern,limit,claim):
    demand(type(pattern) is bytes and 0<len(pattern)<=4097 and integer(limit,256),'query policy')
    root=roots(vault,pin,'GLYPH_BOUNDED_IMPORT_STREAM_V2',100000000)
    decoder=vault/'decoder';demand(sha(read(decoder,1<<20))==root['decoder_sha256'],'decoder identity')
    count=0;coordinates=[];high=False;w=stream.Work()
    with tempfile.TemporaryDirectory(prefix='glyph-stream-proof-') as tmp:
        for d in root['documents']:
            obj=vault/'objects'/d['sha256'];demand(obj.is_dir() and not obj.is_symlink(),'missing object')
            if not d['bytes']:
                demand(read(obj/'empty.json',512)==canon(dict(format='GLYPH_EMPTY_SOURCE_V0',sha256=sha(b''),bytes=0)),'empty binding');continue
            raw=read(obj/'root.json',8192);demand(sha(raw)==d['manifest'],'root pin');r=parse(raw)
            demand(r['source']==dict(bytes=d['bytes'],sha256=d['sha256']),'source identity')
            demand(set(r['artifacts'])=={'rlb2','rlr2','ef'},'index coverage')
            payload={n:w.file(obj/n,r['artifacts'][n],128<<20,n!='ef') for n in ('rlb2','rlr2','ef')}
            from types import SimpleNamespace
            stream.index_structure(SimpleNamespace(rlb2=obj/'rlb2',rlr2=obj/'rlr2',ef=obj/'ef'),r,payload,w)
            dest=Path(tmp)/(str(d['doc_id'])+'.raw')
            cmd=[sys.executable,str(REPO/'experiments/loc_text_ef_v0/trusted_materialize_v0.py'),'--manifest',str(obj/'root.json'),'--root-sha256',d['manifest'],'--decoder',str(decoder),'--output',str(dest),'--threads','2','--mode','rank']
            for n in ('rlb2','rlr2','ef'):cmd+=['--'+n,str(obj/n)]
            subprocess.run(cmd,check=True,capture_output=True,timeout=600)
            scan=stream.scan_authenticated(dest,r['source'],pattern,1<<20,w);count+=scan['count'];high|=scan['count']>256
            coordinates.extend([d['doc_id'],d['revision'],off] for off in scan['offsets'])
    expected=dict(format='GLYPH_STREAM_EXACT_RECEIPT_V2',snapshot=pin,query_hex=pattern.hex(),limit=limit,count=count,coordinates=[] if high else coordinates[:limit],enumeration='OMITTED_HIGH_COUNT' if high else 'PREFIX' if count>limit else 'COMPLETE',status='UNTRUSTED' if high or count>limit else 'FOUND' if count else 'PROVEN_EMPTY',covered_doc_ids=[d['doc_id'] for d in root['documents']],independent_index_source_semantics=False)
    demand(canon(expected)==canon(claim),'independent answer mismatch')
    return dict(status='STREAM_FULL_SOURCE_REPLAY_PASS',count=count,source_read_bytes=w.source_read_bytes,independent_index_source_semantics=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('vault','pin','query-hex','claim'):p.add_argument('--'+n,required=True)
    p.add_argument('--limit',type=int,default=256);a=p.parse_args()
    try:print(json.dumps(verify(Path(a.vault),a.pin,bytes.fromhex(a.query_hex),a.limit,parse(read(Path(a.claim),1<<20))),sort_keys=True))
    except Exception as e:print(json.dumps(dict(status='UNTRUSTED',reason=str(e)[:300])),file=sys.stderr);sys.exit(3)
