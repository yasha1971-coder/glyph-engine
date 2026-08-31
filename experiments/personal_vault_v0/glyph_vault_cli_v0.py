#!/usr/bin/env python3
import argparse,hashlib,json,os,shutil,subprocess,tempfile,time,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha256_path(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def run(*args):
    # Keep stdout reserved for the CLI's single machine-readable JSON result.
    # Child-tool progress and diagnostics go to stderr so callers can safely json.loads(stdout).
    subprocess.check_call([str(x) for x in args],stdout=sys.stderr,stderr=sys.stderr)

def repo_meta(vault):
    p=vault/'repo.meta'
    if not p.is_file(): raise SystemExit('not a GLYPH Vault: '+str(vault))
    return json.loads(p.read_text())

def init(vault):
    vault=vault.resolve()
    if vault.exists() and any(vault.iterdir()): raise SystemExit('vault path must be empty')
    vault.mkdir(parents=True,exist_ok=True)
    for d in ('manifests/roots','manifests/snapshots','segments','objects','derived/text','derived/metadata','derived/ai','journal','cache','quarantine'):
        (vault/d).mkdir(parents=True,exist_ok=True)
    meta={'format':'GLYPH_PERSONAL_VAULT_REPO_V0','created_unix_ns':time.time_ns(),'next_segment_id':1,'source_deletion_enabled':False,'quarantine_required':True}
    (vault/'repo.meta').write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'ok':True,'action':'init','vault':str(vault),'source_deletion_enabled':False},sort_keys=True))

def ensure_build(build_dir):
    sa=build_dir/'build_sa_binary_v1'; bwt=build_dir/'build_bwt_binary_v1'
    if sa.is_file() and bwt.is_file(): return sa,bwt
    run('cmake','-S',ROOT,'-B',build_dir,'-DCMAKE_BUILD_TYPE=Release')
    run('cmake','--build',build_dir,'--target','build_sa_binary_v1','build_bwt_binary_v1','-j2')
    return sa,bwt

def add(vault,source):
    meta=repo_meta(vault); source=source.resolve()
    if not source.is_dir(): raise SystemExit('V0 add accepts a directory')
    seg_id=int(meta['next_segment_id']); name=f'{seg_id:08d}'
    final=vault/'segments'/name
    if final.exists(): raise SystemExit('segment already exists')
    staging=vault/'journal'/('staging-'+name)
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    corpus=staging/'corpus.bin'; objects=staging/'objects.json'; sa=staging/'sa.bin'; bwt=staging/'bwt.bin'; rlb=staging/'bwt.rlb3x'; loc=staging/'locate.loc2'
    run('python3',HERE/'vault_v0.py','pack',source,corpus,objects)
    om=json.loads(objects.read_text()); count=len(om['objects'])
    if count==0: raise SystemExit('source directory contains no files')
    build=vault/'cache'/'build'
    sa_bin,bwt_bin=ensure_build(build)
    run(sa_bin,corpus,sa); run(bwt_bin,corpus,sa,bwt)
    run('python3',HERE/'rlb3x_fixed.py',bwt,rlb,staging/'rlb3x-report.json','--block-runs','8192')
    run('python3',HERE/'loc2_experimental.py','build',sa,str(corpus.stat().st_size+1),'128',loc)
    run('python3',HERE/'loc2_experimental.py','verify',sa,loc)
    restored=staging/'restore-test.bin'
    run('python3',HERE/'restore_rlb3x.py',rlb,restored,staging/'restore-report.json')
    if corpus.read_bytes()!=restored.read_bytes(): raise SystemExit('segment corpus restore mismatch')
    restored_bytes=restored.read_bytes()
    for o in om['objects']:
        blob=restored_bytes[o['offset']:o['offset']+o['bytes']]
        if hashlib.sha256(blob).hexdigest()!=o['sha256']: raise SystemExit('object hash mismatch: '+o['path'])
    source_snapshot=[]
    for o in om['objects']:
        p=source/o['path']; st=p.stat()
        source_snapshot.append({'id':o['id'],'path':o['path'],'bytes':o['bytes'],'sha256':o['sha256'],'source_mtime_ns':st.st_mtime_ns})
    source_state={'format':'GLYPH_VAULT_SOURCE_STATE_V0','source_root':str(source),'objects':source_snapshot}
    (staging/'source-state.json').write_text(json.dumps(source_state,sort_keys=True,separators=(',',':'))+'\n')
    manifest={'format':'GLYPH_VAULT_SEGMENT_MANIFEST_V0','segment_id':name,'created_unix_ns':time.time_ns(),'source_root':str(source),'object_count':count,'source_bytes':corpus.stat().st_size,'files':{}}
    for key,p in {'rlb3x':rlb,'loc2':loc,'objects':objects,'source_state':staging/'source-state.json'}.items():
        manifest['files'][key]={'name':p.name,'bytes':p.stat().st_size,'sha256':sha256_path(p)}
    manifest['restore_tested']=True; manifest['object_hashes_verified']=True; manifest['eligible_to_free_source']=False
    (staging/'segment-manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
    for p in (corpus,sa,bwt,restored,staging/'rlb3x-report.json',staging/'restore-report.json'):
        if p.exists(): p.unlink()
    os.replace(staging,final)
    root_manifest={'format':'GLYPH_VAULT_ROOT_MANIFEST_V0','committed_unix_ns':time.time_ns(),'segments':[p.name for p in sorted((vault/'segments').iterdir()) if p.is_dir()]}
    root_tmp=vault/'journal'/'root.tmp'; root_tmp.write_text(json.dumps(root_manifest,sort_keys=True,separators=(',',':'))+'\n')
    root_name=f'{time.time_ns()}.json'; os.replace(root_tmp,vault/'manifests/roots'/root_name)
    meta['next_segment_id']=seg_id+1; (vault/'repo.meta').write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'ok':True,'action':'add','segment_id':name,'objects':count,'source_bytes':manifest['source_bytes'],'published':True,'source_deleted':False},sort_keys=True))

def iter_segments(vault):
    repo_meta(vault)
    return [p for p in sorted((vault/'segments').iterdir()) if p.is_dir()]

def verify_segment(seg):
    m=json.loads((seg/'segment-manifest.json').read_text())
    for info in m['files'].values():
        p=seg/info['name']
        if not p.is_file() or p.stat().st_size!=info['bytes'] or sha256_path(p)!=info['sha256']:
            raise SystemExit('segment file verification failed: '+str(p))
    with tempfile.TemporaryDirectory(prefix='glyph-verify-') as td:
        restored=Path(td)/'restored.bin'
        run('python3',HERE/'restore_rlb3x.py',seg/'bwt.rlb3x',restored,Path(td)/'report.json')
        data=restored.read_bytes(); om=json.loads((seg/'objects.json').read_text())
        for o in om['objects']:
            blob=data[o['offset']:o['offset']+o['bytes']]
            if len(blob)!=o['bytes'] or hashlib.sha256(blob).hexdigest()!=o['sha256']:
                raise SystemExit('object verification failed: '+o['path'])
    return len(om['objects']),sum(o['bytes'] for o in om['objects'])

def verify(vault):
    total_obj=total_bytes=0
    segs=iter_segments(vault)
    for seg in segs:
        n,b=verify_segment(seg); total_obj+=n; total_bytes+=b
    report={'ok':True,'action':'verify','segments':len(segs),'objects':total_obj,'recoverable_bytes':total_bytes,'full_restore_hash_check':True}
    (vault/'manifests'/'last-verify.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps(report,sort_keys=True))

def find_object(vault,selector):
    matches=[]
    for seg in iter_segments(vault):
        om=json.loads((seg/'objects.json').read_text())
        for o in om['objects']:
            if selector==str(o['id']) or selector==o['path']:
                matches.append((seg,o))
    if not matches: raise SystemExit('object not found: '+selector)
    if len(matches)>1 and selector.isdigit(): raise SystemExit('numeric object id is ambiguous across segments; use path')
    if len(matches)>1: raise SystemExit('path exists in multiple segments; version selection not implemented in V0')
    return matches[0]

def restore(vault,selector,out):
    seg,o=find_object(vault,selector)
    with tempfile.TemporaryDirectory(prefix='glyph-restore-') as td:
        corpus=Path(td)/'corpus.bin'; run('python3',HERE/'restore_rlb3x.py',seg/'bwt.rlb3x',corpus,Path(td)/'report.json')
        with corpus.open('rb') as f:
            f.seek(o['offset']); blob=f.read(o['bytes'])
    if hashlib.sha256(blob).hexdigest()!=o['sha256']: raise SystemExit('restored object hash mismatch')
    out=out.resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(blob)
    print(json.dumps({'ok':True,'action':'restore','path':o['path'],'bytes':len(blob),'sha256':o['sha256'],'output':str(out)},sort_keys=True))

def free_space(vault):
    verified=(vault/'manifests'/'last-verify.json').is_file()
    eligible=[]; changed=[]; missing=[]
    for seg in iter_segments(vault):
        ss=json.loads((seg/'source-state.json').read_text()); root=Path(ss['source_root'])
        for o in ss['objects']:
            p=root/o['path']
            if not p.exists(): missing.append(o); continue
            if not p.is_file() or p.stat().st_size!=o['bytes'] or sha256_path(p)!=o['sha256']:
                changed.append(o); continue
            if verified: eligible.append(o)
    report={'ok':True,'action':'free-space','dry_run':True,'source_deletion_performed':False,'full_verify_record_present':verified,'eligible_objects':len(eligible),'safe_to_free_bytes':sum(x['bytes'] for x in eligible),'changed_objects':len(changed),'missing_source_objects':len(missing),'policy':'V0 reports eligibility only; permanent source deletion is disabled.'}
    print(json.dumps(report,sort_keys=True))

def list_objects(vault):
    rows=[]
    for seg in iter_segments(vault):
        om=json.loads((seg/'objects.json').read_text())
        for o in om['objects']: rows.append({'segment':seg.name,'id':o['id'],'path':o['path'],'bytes':o['bytes'],'sha256':o['sha256']})
    print(json.dumps({'format':'GLYPH_VAULT_LIST_V0','objects':rows},sort_keys=True,separators=(',',':')))

def main():
    ap=argparse.ArgumentParser(prog='glyph-vault-v0'); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('init'); p.add_argument('vault',type=Path)
    p=sp.add_parser('add'); p.add_argument('vault',type=Path); p.add_argument('source',type=Path)
    p=sp.add_parser('verify'); p.add_argument('vault',type=Path)
    p=sp.add_parser('list'); p.add_argument('vault',type=Path)
    p=sp.add_parser('restore'); p.add_argument('vault',type=Path); p.add_argument('selector'); p.add_argument('out',type=Path)
    p=sp.add_parser('free-space'); p.add_argument('vault',type=Path); p.add_argument('--dry-run',action='store_true',required=True)
    a=ap.parse_args()
    if a.cmd=='init': init(a.vault)
    elif a.cmd=='add': add(a.vault,a.source)
    elif a.cmd=='verify': verify(a.vault)
    elif a.cmd=='list': list_objects(a.vault)
    elif a.cmd=='restore': restore(a.vault,a.selector,a.out)
    elif a.cmd=='free-space': free_space(a.vault)

if __name__=='__main__': main()
