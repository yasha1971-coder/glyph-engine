#!/usr/bin/env python3
import argparse,json,os,shutil,time
from pathlib import Path

import glyph_vault_cli_v0 as base
import real_intake_v1

HERE=Path(__file__).resolve().parent


def add_v1(vault,source):
    meta=base.repo_meta(vault); source=source.resolve()
    if not source.is_dir(): raise SystemExit('V1 add accepts a directory')
    seg_id=int(meta['next_segment_id']); name=f'{seg_id:08d}'
    final=vault/'segments'/name
    if final.exists(): raise SystemExit('segment already exists')
    staging=vault/'journal'/('staging-'+name)
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)

    corpus=staging/'corpus.bin'; objects=staging/'objects.json'; intake=staging/'intake.json'
    sa=staging/'sa.bin'; bwt=staging/'bwt.bin'; rlb=staging/'bwt.rlb3x'; loc=staging/'locate.loc2'
    intake_doc=real_intake_v1.pack(source,corpus,objects,intake)
    om=json.loads(objects.read_text()); count=len(om['objects'])
    if count==0: raise SystemExit('source directory contains no files')

    build=vault/'cache'/'build'
    sa_bin,bwt_bin=base.ensure_build(build)
    base.run(sa_bin,corpus,sa); base.run(bwt_bin,corpus,sa,bwt)
    base.run('python3',HERE/'rlb3x_fixed.py',bwt,rlb,staging/'rlb3x-report.json','--block-runs','8192')
    base.run('python3',HERE/'loc2_experimental.py','build',sa,str(corpus.stat().st_size+1),'128',loc)
    base.run('python3',HERE/'loc2_experimental.py','verify',sa,loc)

    restored=staging/'restore-test.bin'
    base.run('python3',HERE/'restore_rlb3x.py',rlb,restored,staging/'restore-report.json')
    if corpus.read_bytes()!=restored.read_bytes(): raise SystemExit('segment corpus restore mismatch')
    restored_bytes=restored.read_bytes()
    for o in om['objects']:
        blob=restored_bytes[o['offset']:o['offset']+o['bytes']]
        if real_intake_v1.sha_bytes(blob)!=o['sha256']: raise SystemExit('object hash mismatch: '+o['path'])

    metadata_by_path={x['path']:x for x in intake_doc['files']}
    source_snapshot=[]
    for o in om['objects']:
        md=metadata_by_path[o['path']]
        source_snapshot.append({
            'id':o['id'],'path':o['path'],'bytes':o['bytes'],'sha256':o['sha256'],
            'source_mtime_ns':md['mtime_ns'],'logical_version_id':md['logical_version_id'],
            'content_id':md['content_id'],'mime_guess':md['mime_guess'],
        })
    source_state={'format':'GLYPH_VAULT_SOURCE_STATE_V1','source_root':str(source),'objects':source_snapshot}
    (staging/'source-state.json').write_text(json.dumps(source_state,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')

    manifest={
        'format':'GLYPH_VAULT_SEGMENT_MANIFEST_V1','segment_id':name,'created_unix_ns':time.time_ns(),
        'source_root':str(source),'object_count':count,'source_bytes':corpus.stat().st_size,
        'intake_format':intake_doc['format'],'directory_count':intake_doc['directory_count'],'files':{},
    }
    for key,p in {'rlb3x':rlb,'loc2':loc,'objects':objects,'source_state':staging/'source-state.json','intake':intake}.items():
        manifest['files'][key]={'name':p.name,'bytes':p.stat().st_size,'sha256':base.sha256_path(p)}
    manifest['restore_tested']=True; manifest['object_hashes_verified']=True
    manifest['filesystem_metadata_captured']=True; manifest['eligible_to_free_source']=False
    (staging/'segment-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')

    for p in (corpus,sa,bwt,restored,staging/'rlb3x-report.json',staging/'restore-report.json'):
        if p.exists(): p.unlink()
    prior=base.committed_segment_ids(vault)
    os.replace(staging,final)
    root_name=base.publish_root(vault,prior+[name])
    meta['next_segment_id']=seg_id+1
    meta['latest_intake_format']='GLYPH_REAL_INTAKE_V1'
    (vault/'repo.meta').write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({
        'ok':True,'action':'add','intake_format':'GLYPH_REAL_INTAKE_V1','segment_id':name,
        'objects':count,'directories':intake_doc['directory_count'],'source_bytes':manifest['source_bytes'],
        'published':True,'root_manifest':root_name,'source_deleted':False
    },sort_keys=True))


def main():
    ap=argparse.ArgumentParser(prog='glyph-vault-v1'); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('init'); p.add_argument('vault',type=Path)
    p=sp.add_parser('add'); p.add_argument('vault',type=Path); p.add_argument('source',type=Path)
    p=sp.add_parser('verify'); p.add_argument('vault',type=Path)
    p=sp.add_parser('list'); p.add_argument('vault',type=Path)
    p=sp.add_parser('restore'); p.add_argument('vault',type=Path); p.add_argument('selector'); p.add_argument('out',type=Path)
    p=sp.add_parser('free-space'); p.add_argument('vault',type=Path); p.add_argument('--dry-run',action='store_true',required=True)
    a=ap.parse_args()
    if a.cmd=='init': base.init(a.vault)
    elif a.cmd=='add': add_v1(a.vault,a.source)
    elif a.cmd=='verify': base.verify(a.vault)
    elif a.cmd=='list': base.list_objects(a.vault)
    elif a.cmd=='restore': base.restore(a.vault,a.selector,a.out)
    elif a.cmd=='free-space': base.free_space(a.vault)

if __name__=='__main__': main()
