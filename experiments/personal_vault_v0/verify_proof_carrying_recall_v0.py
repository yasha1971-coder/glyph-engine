#!/usr/bin/env python3
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESTORE=HERE/'restore_rlb3x.py'


def sha256_path(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def fail(msg):
    raise SystemExit('VERIFY_FAIL: '+msg)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('vault',type=Path)
    ap.add_argument('receipt',type=Path)
    a=ap.parse_args()
    r=json.loads(a.receipt.read_text())
    if r.get('format')!='GLYPH_PROOF_CARRYING_RECALL_V0': fail('bad receipt format')
    root=a.vault/'manifests'/'roots'/r['vault_commit']['root_name']
    if not root.is_file(): fail('committed root missing')
    if sha256_path(root)!=r['vault_commit']['root_sha256']: fail('root hash mismatch')
    root_doc=json.loads(root.read_text())
    sid=r['segment']['segment_id']
    entries={e['segment_id']:e for e in root_doc.get('segment_entries',[])}
    e=entries.get(sid)
    if e is None: fail('segment not bound by root')
    if e['segment_manifest_sha256']!=r['segment']['segment_manifest_sha256']: fail('receipt/root segment hash disagreement')
    seg=a.vault/'segments'/sid
    manifest=seg/'segment-manifest.json'
    if not manifest.is_file() or sha256_path(manifest)!=e['segment_manifest_sha256']: fail('segment manifest hash mismatch')
    md=json.loads(manifest.read_text())
    for info in md.get('files',{}).values():
        p=seg/info['name']
        if not p.is_file() or p.stat().st_size!=info['bytes'] or sha256_path(p)!=info['sha256']:
            fail('runtime file binding failed: '+str(p))
    om=json.loads((seg/'objects.json').read_text())
    ro=r['object']
    matches=[o for o in om['objects'] if o['id']==ro['object_id'] and o['path']==ro['path']]
    if len(matches)!=1: fail('object identity not unique in object map')
    o=matches[0]
    for k in ('bytes','sha256'):
        if o[k]!=ro[k]: fail('object '+k+' mismatch')
    if o['offset']!=ro['segment_corpus_offset']: fail('object corpus offset mismatch')
    with tempfile.TemporaryDirectory(prefix='glyph-proof-verify-') as td:
        restored=Path(td)/'corpus.bin'
        report=Path(td)/'restore.json'
        subprocess.check_call(['python3',str(RESTORE),str(seg/'bwt.rlb3x'),str(restored),str(report)],stdout=subprocess.DEVNULL)
        with restored.open('rb') as f:
            f.seek(o['offset']); blob=f.read(o['bytes'])
    if len(blob)!=o['bytes']: fail('restored object length mismatch')
    if hashlib.sha256(blob).hexdigest()!=o['sha256']: fail('restored object sha256 mismatch')
    pattern=bytes.fromhex(r['claim']['pattern_hex'])
    if pattern.decode('utf-8')!=r['claim']['pattern_utf8']: fail('pattern utf8/hex disagreement')
    offsets=r['evidence']['object_offsets']
    if len(offsets)!=r['evidence']['occurrences']: fail('occurrence count mismatch')
    for off in offsets:
        if off<0 or off+len(pattern)>len(blob): fail('evidence offset outside object')
        if blob[off:off+len(pattern)]!=pattern: fail('literal absent at claimed offset')
    # Ensure receipt did not omit extra literal occurrences inside this object.
    actual=[]; pos=0
    while True:
        pos=blob.find(pattern,pos)
        if pos<0: break
        actual.append(pos); pos+=1
    if actual!=offsets: fail('receipt occurrence set incomplete or incorrect')
    print(json.dumps({
      'ok':True,
      'format':'GLYPH_PROOF_CARRYING_RECALL_VERIFY_V0',
      'root_sha256':r['vault_commit']['root_sha256'],
      'segment_id':sid,
      'path':o['path'],
      'object_sha256':o['sha256'],
      'pattern_utf8':r['claim']['pattern_utf8'],
      'occurrences':len(actual),
      'original_source_required':False,
      'ai_required':False,
      'verified_from_committed_glyph_state':True
    },ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
