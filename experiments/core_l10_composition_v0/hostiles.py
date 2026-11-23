#!/usr/bin/env python3
"""Named black-box hostile cases; independent verifier rejects each, no mocks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILE = 'GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0'


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def ident(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=sha(raw))


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle','receipts','oracles','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir()
    copied = a.output/'independent.py'
    shutil.copy2(HERE/'verifier.py', copied)
    target = a.output/'receipt.json'
    records = []

    def invoke(name, receipt=None, snapshot='A1', case='q0', changes=None, raw=None):
        root = a.bundle/'roots'/(snapshot+'.json')
        r = json.loads((a.receipts/snapshot/(case+'.json')).read_bytes()) if receipt is None else receipt
        target.write_bytes(raw if raw is not None else canonical(r))
        params = dict(bundle=a.bundle,root=root,pin=sha(root.read_bytes()),generation=int(snapshot[1]),oracles=a.oracles,
            receipt=target,query_hex=r['query_hex'],limit=r['max_offsets'],research=REPO,reference=a.reference)
        if snapshot[1]=='1':
            old = root.with_name(snapshot[0]+'0.json')
            params.update(previous_root=old,previous_pin=sha(old.read_bytes()))
        params.update(changes or {})
        args=[sys.executable,'-I',str(copied)]
        for k,v in params.items():
            args += ['--'+k.replace('_','-'),str(v)]
        result=subprocess.run(args,capture_output=True,text=True,timeout=90)
        assert result.returncode==3, (name,result.stdout,result.stderr)
        info=json.loads(result.stderr)
        assert info['status']=='UNTRUSTED'
        records.append(dict(name=name,exit_code=3,result=info))
        return r,params

    baseline=json.loads((a.receipts/'A1/q0.json').read_bytes())
    def mutate(name,edit,case='q0',snapshot='A1'):
        r=json.loads((a.receipts/snapshot/(case+'.json')).read_bytes())
        edit(r)
        invoke(name,r,snapshot,case)
    def pool(name,edit,snapshot='A1',case='q0'):
        b=a.output/name
        shutil.copytree(a.bundle,b)
        edit(b)
        invoke(name,snapshot=snapshot,case=case,changes=dict(bundle=b))
    pool('missing-expected-block',lambda b:(b/'blocks'/(json.loads((b/'roots/A1.json').read_bytes())['blocks'][0]['manifest']['sha256']+'.json')).unlink())
    pool('missing-required-leaf',lambda b:shutil.rmtree(b/'leaves/3'))
    pool('missing-empty-leaf-binding',lambda b:(b/'leaves/0/binding.json').unlink())
    pool('duplicate-leaf-substituted',lambda b:shutil.copyfile(b/'leaves/5/binding.json',b/'leaves/6/binding.json'))
    pool('stale-leaf-in-V1-slot',lambda b:shutil.copyfile(b/'leaves/4/binding.json',b/'leaves/9/binding.json'))
    def flip_old(b):
        f=b/'leaves/3/rlb2'
        raw=bytearray(f.read_bytes());raw[-1]^=1;f.write_bytes(raw)
    pool('old-leaf-bytes-changed-after-append',flip_old)
    mutate('changed-ordered-block-coverage',lambda r:r['coverage']['queried_blocks'].reverse())
    mutate('identical-documents-merged',lambda r:r['leaf_results'].pop(6),case='q1')
    mutate('empty-document-removed',lambda r:r['leaf_results'].pop(0))
    mutate('false-empty-with-incomplete-V1-coverage',lambda r:(r.update(match_count=0,status='PROVEN_EMPTY',coordinates=[],excerpts=[],returned_count=0),r['coverage']['queried_leaves'].pop()),case='q9')
    mutate('forged-new-version-label',lambda r:r.update(generation=1),snapshot='A0')
    invoke('wrong-expected-V0-parent-pin',changes=dict(previous_pin='0'*64))
    invoke('missing-separately-pinned-parent',changes=dict(previous_root=a.output/'missing'))
    mutate('unacknowledged-new-V1-leaf',lambda r:r['leaf_results'].pop(9),case='q9')
    def fake_hit(r,coordinate):
        r.update(match_count=1,returned_count=1,coordinates=[coordinate],status='FOUND')
    mutate('cross-document-concatenation-hit',lambda r:fake_hit(r,[3,13]),case='q13',snapshot='A0')
    mutate('cross-block-concatenation-hit',lambda r:fake_hit(r,[5,0]),case='q14',snapshot='A0')
    mutate('fabricated-global-doc-ID',lambda r:r['coordinates'][0].__setitem__(0,999))
    def perblock(r):
        r.update(coordinates=[[3,7],[7,5]],returned_count=2)
    mutate('cap-applied-independently-per-block',perblock,case='cap1',snapshot='A0')
    mutate('bad-excerpt-SHA',lambda r:r['excerpts'][0].update(sha256='0'*64))
    mutate('wrong-source-identity',lambda r:r['leaf_results'][3]['receipt']['source'].update(sha256='0'*64))
    invoke('wrong-external-root-pin',changes=dict(pin='0'*64))
    mutate('partial-block-result-count',lambda r:r['block_counts'].pop())
    mutate('count-overflow-attempt',lambda r:r.update(match_count=2**64))
    mutate('unsorted-coordinates',lambda r:r['coordinates'].reverse())
    mutate('duplicate-coordinate',lambda r:r['coordinates'].append(r['coordinates'][0]))
    mutate('missing-positive-coordinate',lambda r:r['coordinates'].pop())
    mutate('stored-success-not-recomputed',lambda r:r['leaf_results'][3]['receipt']['excerpts'][0].update(sha256='0'*64))
    invoke('changed-expected-query',changes=dict(query_hex=b'other'.hex()))
    invoke('old-V0-receipt-replayed-as-V1',json.loads((a.receipts/'A0/q9.json').read_bytes()),case='q9')
    invoke('V0-receipt-against-other-layout',json.loads((a.receipts/'A0/q0.json').read_bytes()),snapshot='B0')
    mutate('unknown-authority-claim-field',lambda r:r.update(signed_current_world=True))
    mutate('bool-numeric-count',lambda r:r.update(match_count=True))
    raw=canonical(baseline)
    for name,bad in [('duplicate-JSON-key',b'{"match_count":0,'+raw[1:]),('truncated-receipt',raw[:-7]),
                     ('noncanonical-receipt',b' '+raw),('invalid-UTF8',raw[:-2]+b'\xff}\n')]:
        invoke(name,raw=bad)
    invoke('empty-query',changes=dict(query_hex=''))
    # Re-pin the COMPLETE physical graph, not just EF: source identities and
    # geometry remain valid, select checkpoint is malformed. Use generation 0
    # so the intended physical validator, rather than append-prefix policy,
    # must discover the original L9 falsifier.
    b=a.output/'repinned-EF'
    shutil.copytree(a.bundle,b)
    leaf=b/'leaves/1'
    ef=bytearray((leaf/'ef').read_bytes())
    old=struct.unpack_from('<I',ef,len(ef)-4)[0]
    struct.pack_into('<I',ef,len(ef)-4,old+1)
    (leaf/'ef').write_bytes(ef)
    lr=json.loads((leaf/'root.json').read_bytes())
    lr['artifacts']['ef']=ident(leaf/'ef')
    (leaf/'root.json').write_bytes(canonical(lr))
    binding=json.loads((leaf/'binding.json').read_bytes())
    binding.update(index_root=ident(leaf/'root.json'),artifacts=lr['artifacts'])
    (leaf/'binding.json').write_bytes(canonical(binding))
    root=json.loads((b/'roots/A0.json').read_bytes())
    root['leaves'][1]['binding']=ident(leaf/'binding.json')
    block=root['blocks'][0]
    phys=json.loads((b/'blocks'/(block['manifest']['sha256']+'.json')).read_bytes())
    phys['leaves']=root['leaves'][:3]
    path=b/'blocks'/(sha(canonical(phys))+'.json')
    path.write_bytes(canonical(phys))
    block['manifest']=ident(path)
    root.pop('research_composition_root_id')
    root['research_composition_root_id']=sha(PROFILE.encode()+b'\0ROOT\0'+canonical(root))
    rp=b/'roots/A0.json';rp.write_bytes(canonical(root))
    r=json.loads((a.receipts/'A0/q0.json').read_bytes())
    r.update(root_sha256=sha(rp.read_bytes()),research_composition_root_id=root['research_composition_root_id'])
    r['leaf_results'][1]['binding_sha256']=root['leaves'][1]['binding']['sha256']
    primitive=r['leaf_results'][1]['receipt']
    primitive.update(trusted_root_sha256=binding['index_root']['sha256'],artifacts=lr['artifacts'])
    for name in ('expected_leaves','verified_leaves','queried_leaves'):
        r['coverage'][name]=[x['binding']['sha256'] for x in root['leaves']]
    for name in ('expected_blocks','verified_blocks','queried_blocks'):
        r['coverage'][name]=[x['manifest']['sha256'] for x in root['blocks']]
    invoke('physically-repinned-malformed-EF-select-checkpoint',r,snapshot='A0',changes=dict(bundle=b,root=rp,pin=sha(rp.read_bytes())))
    assert 'EF physical samples/geometry' in records[-1]['result']['reason'], records[-1]
    cp=subprocess.run([sys.executable,'-I',str(HERE/'producer.py'),'--bundle',str(b),'--root',str(rp),
        '--pin',sha(rp.read_bytes()),'--generation','0','--reference',str(a.reference),'--query-hex',b'not-present'.hex()],
        capture_output=True,text=True,timeout=90)
    assert cp.returncode==3 and json.loads(cp.stderr)['status']=='UNTRUSTED' and 'EF select checkpoint' in cp.stderr
    records[-1]['producer_exit_code']=cp.returncode
    result=dict(status='HOSTILE_COMPOSITION_PASS',hostile_rejections=len(records),cases=records,
                repinned_EF_producer_rejected=True, no_composition_producer_import_in_verifier=True)
    assert len(records)>=25
    (a.output/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
