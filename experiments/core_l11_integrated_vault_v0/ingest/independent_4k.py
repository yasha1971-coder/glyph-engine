"""Independent 4K full-restore + source scan + L9 physical-oracle replay.
No importer or producer imports. Internal physical-source oracle remains quadratic.
"""
import argparse, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path
from independent_verify import roots,read,parse,sha,canon,demand,integer,REPO

def verify(vault,pin,pattern,limit,claim,selected=None,output=None):
    demand(type(pattern) is bytes and 0<len(pattern)<=4097 and integer(limit,256),'query admission')
    root=roots(vault,pin,'GLYPH_BOUNDED_IMPORT_4K_V1',4096)
    decoder=vault/'decoder'; demand(sha(read(decoder,1<<20))==root['decoder_sha256'],'decoder identity')
    spec=importlib.util.spec_from_file_location('independent_l9_physical',REPO/'experiments/core_l9_exact_query_v0/verifier.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    coords=[]; replays=0; picked=None
    with tempfile.TemporaryDirectory(prefix='glyph-4k-independent-') as tmp:
        tmp=Path(tmp)
        for d in root['documents']:
            obj=vault/'objects'/d['sha256'];demand(obj.is_dir() and not obj.is_symlink(),'missing/linked object')
            raw=b''
            if d['bytes']:
                manifest=read(obj/'root.json',8192);demand(sha(manifest)==d['manifest'],'manifest pin')
                physical=parse(manifest); demand(physical['source']=={'bytes':d['bytes'],'sha256':d['sha256']},'source binding')
                dest=tmp/(str(d['doc_id'])+'.raw')
                cmd=[sys.executable,str(REPO/'experiments/loc_text_ef_v0/trusted_materialize_v0.py'),'--manifest',str(obj/'root.json'),'--root-sha256',d['manifest'],'--decoder',str(decoder),'--output',str(dest),'--threads','2','--mode','rank']
                for n in ('rlb2','rlr2','ef'): cmd+=['--'+n,str(obj/n)]
                subprocess.run(cmd,check=True,capture_output=True,timeout=90)
                raw=read(dest,4096)
                demand(set(physical['artifacts'])=={'rlb2','rlr2','ef'},'physical coverage')
                components={n:read(obj/n,1<<20) for n in ('rlb2','rlr2','ef')}
                for n,data in components.items(): demand(sha(data)==physical['artifacts'][n]['sha256'] and len(data)==physical['artifacts'][n]['bytes'],'component binding')
                oracle.physical_oracle(raw,components,oracle.Work());replays+=1
            else: demand(read(obj/'empty.json',512)==canon({'format':'GLYPH_EMPTY_SOURCE_V0','sha256':sha(b''),'bytes':0}),'empty binding')
            demand(len(raw)==d['bytes'] and sha(raw)==d['sha256'],'restored content')
            coords.extend([d['doc_id'],d['revision'],i] for i in range(len(raw)) if raw.startswith(pattern,i))
            if d['doc_id']==selected: picked=raw
    expected=dict(format='GLYPH_4K_EXACT_RECEIPT_V1',snapshot=pin,vault_id=root['vault_id'],query_hex=pattern.hex(),limit=limit,count=len(coords),coordinates=coords[:limit],covered_doc_ids=[d['doc_id'] for d in root['documents']],coverage_complete=True,enumeration_complete=len(coords)<=limit,status='UNTRUSTED' if len(coords)>limit else 'FOUND' if coords else 'PROVEN_EMPTY')
    demand(canon(expected)==canon(claim),'logical answer/coverage/policy mismatch')
    if selected is not None:
        demand(picked is not None and output is not None,'selection absent')
        with Path(output).open('xb') as f:f.write(picked)
    return dict(status='INDEPENDENT_4K_FULL_REPLAY_PASS',documents=len(root['documents']),physical_oracles=replays,count=len(coords),enumeration_complete=expected['enumeration_complete'])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('vault','pin','query-hex','claim'):p.add_argument('--'+n,required=True)
    p.add_argument('--limit',type=int,default=256);p.add_argument('--selected',type=int);p.add_argument('--output')
    a=p.parse_args()
    try:print(json.dumps(verify(Path(a.vault),a.pin,bytes.fromhex(a.query_hex),a.limit,parse(read(Path(a.claim),1<<20)),a.selected,a.output),sort_keys=True))
    except Exception as e:print(json.dumps({'status':'UNTRUSTED','reason':str(e)[:300]}),file=sys.stderr);sys.exit(3)
