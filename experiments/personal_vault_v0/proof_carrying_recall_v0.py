#!/usr/bin/env python3
import argparse,hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
QUERY=HERE/'query_rlb3x_object.py'


def sha256_path(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def latest_root(vault):
    roots=sorted((vault/'manifests'/'roots').glob('*.json'))
    if not roots: raise SystemExit('vault has no committed root')
    p=roots[-1]
    return p,json.loads(p.read_text())


def query_segment(seg,pattern):
    out=subprocess.check_output([
        'python3',str(QUERY),
        '--rlb3x',str(seg/'bwt.rlb3x'),
        '--locate-core',str(seg/'locate.loc2'),
        '--objects',str(seg/'objects.json'),
        '--pattern-hex',pattern.hex(),
    ],text=True)
    return json.loads(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('vault',type=Path)
    ap.add_argument('--pattern',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    pattern=a.pattern.encode('utf-8')
    root_path,root=latest_root(a.vault)
    entries={e['segment_id']:e for e in root.get('segment_entries',[])}
    hits=[]
    for sid in root.get('segments',[]):
        seg=a.vault/'segments'/sid
        got=query_segment(seg,pattern)
        for h in got.get('valid_hits',[]):
            hits.append((seg,h))
    objects={(seg.name,h['object_id'],h['path']) for seg,h in hits}
    if len(objects)!=1:
        raise SystemExit('proof receipt requires exactly one matching object version; got '+str(len(objects)))
    sid,oid,path=next(iter(objects))
    seg=a.vault/'segments'/sid
    om=json.loads((seg/'objects.json').read_text())
    obj=next(o for o in om['objects'] if o['id']==oid and o['path']==path)
    object_hits=sorted(h['object_offset'] for s,h in hits if s.name==sid and h['object_id']==oid and h['path']==path)
    entry=entries.get(sid)
    if entry is None: raise SystemExit('latest root lacks segment binding')
    receipt={
      'format':'GLYPH_PROOF_CARRYING_RECALL_V0',
      'claim':{'type':'literal_present_in_object','pattern_utf8':a.pattern,'pattern_hex':pattern.hex()},
      'vault_commit':{
        'root_name':root_path.name,
        'root_sha256':sha256_path(root_path),
        'parent_root_name':root.get('parent_root_name'),
        'parent_root_sha256':root.get('parent_root_sha256'),
      },
      'segment':{
        'segment_id':sid,
        'segment_manifest_sha256':entry['segment_manifest_sha256'],
      },
      'object':{
        'object_id':oid,'path':path,'bytes':obj['bytes'],'sha256':obj['sha256'],
        'segment_corpus_offset':obj['offset'],
      },
      'evidence':{
        'object_offsets':object_hits,
        'occurrences':len(object_hits),
      },
      'verification_contract':[
        'root hash must match receipt',
        'root must bind the named segment manifest hash',
        'segment manifest must bind runtime files',
        'object must restore from GLYPH without source file',
        'restored object SHA-256 must match receipt',
        'pattern bytes must occur at every claimed object offset',
      ],
      'important_non_claim':'This proves committed byte provenance and reproducibility of the literal claim. It does not prove semantic truth, authorship, or that an AI used the evidence correctly.'
    }
    a.out.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'ok':True,'receipt':str(a.out),'path':path,'occurrences':len(object_hits)},ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
