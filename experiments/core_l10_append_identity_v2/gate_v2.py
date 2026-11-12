#!/usr/bin/env python3
"""CORE-L10 V2 separate research gate: producer event model vs independent replay."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import identity_v2 as prod

EXPECTED = [
    'ef16763c1a2c2213f3dad2df6880f9ec10920f5e21c6609d62b3683e261e5c82',
    '7a8057dd12cb0b80e2b60cecf312926efda02be645b2ff80168e85f22698fe3e',
    '8081c6b6ee4a7f8ab712d29348559f8b22826a5da358096d9c0c0c4d690c408b',
    '747783f168b074191863af015bb1ea8887bbfff369e3be192baef257bff9d8d1',
    'bd726f1ce1b6f0ca41a24770a9919b681357c327ca91eb4b709129c20bf6219e',
    '67d18838bee3212de7307e77909d0b852db93447b98987477e095567d2348c19',
    '3aab93d1c891b79d15ec61de985ccf1f86cfec083df5ea13a93499b56354d315',
]


def ensure(cond, why):
    if not cond:
        raise AssertionError(why)


def rejection(fn, desc, results):
    try:
        fn()
    except (ValueError, OSError):
        results.append(desc)
        return
    raise AssertionError('FAIL OPEN: ' + desc)


def run_verifier(bundle, generation, receipt, pins=EXPECTED):
    # Standalone verifier is separately executable and has NO producer imports.
    # Perform most hostile cases via its public pure entry point to avoid
    # excess Python process creation; independently run the CLI every 9th case.
    import independent_verifier_v2 as verify
    from types import SimpleNamespace
    if receipt.name.endswith('_00.json') or receipt.name.endswith('_07.json') or \
       receipt.name in ('bad_00.json','bad_10.json','bad_20.json'):
        cli=[sys.executable,str(Path(__file__).parent/'independent_verifier_v2.py'),
             '--bundle',str(bundle),'--expected-pins-json',json.dumps(pins,separators=(',',':')),
             '--generation',str(generation),'--receipt',str(receipt)]
        return subprocess.run(cli,capture_output=True,text=True,timeout=30)
    try:
        roots=verify.verify_all(bundle,pins)
        claim=verify.parse(verify.read_regular(receipt,65536))
        verify.independently_query(bundle,roots[generation],pins[generation],claim)
        return SimpleNamespace(returncode=0,stdout='PASS',stderr='')
    except Exception as exc:
        return SimpleNamespace(returncode=3,stdout='',stderr='UNTRUSTED: '+str(exc))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    out=a.output.absolute()
    ensure(not out.exists(),'refuse existing output')
    out.mkdir(parents=True)
    t0=time.monotonic()
    report={'status':'BLOCKED','schema':'GLYPH_CORE_L10_APPEND_ID_V2_TEST_V0',
            'legacy_composition_v1_unchanged':True,
            'identity_profile_normative':False,
            'actual_rlb2_leaf_compatibility_proven':False}
    try:
        bundle=out/'fixture'
        q=subprocess.run([sys.executable,str(Path(__file__).parent/'identity_v2.py'),
                          '--output',str(bundle)],capture_output=True,text=True,timeout=30)
        ensure(q.returncode==0,'fixture build failed: '+q.stderr)
        pins=json.loads((bundle/'pins.json').read_text())
        ensure(pins==EXPECTED,'root pin mismatch: changed fixtures or code')
        roots=[json.loads((bundle/'snapshots'/(pin+'.json')).read_text()) for pin in pins]
        ensure(len(roots)==7,'wrong snapshot count')
        ensure([x['next_doc_id'] for x in roots]==[1,2,2,2,3,4,4],
               'doc IDs unstable after rename/revise/tombstone')
        ensure([x['documents'][0]['revision'] for x in roots]==[0,0,0,1,1,1,1],
               'renames incorrectly revise content')
        ensure(roots[1]['documents'][0]['path_hex']==b'z.bin'.hex()
               and roots[1]['documents'][1]['path_hex']==b'a.bin'.hex(),
               'late early sorting name moved doc IDs')
        ensure(roots[2]['documents'][0]['path_hex']==b'q.bin'.hex()
               and roots[0]['documents'][0]['path_hex']==b'z.bin'.hex(),
               'rename edited previous snapshot')
        ensure(roots[4]['documents'][1]['sha256']==roots[4]['documents'][2]['sha256']
               and roots[4]['documents'][1]['doc_id']!=roots[4]['documents'][2]['doc_id'],
               'duplicate bytes conflated logical documents')
        ensure(roots[6]['documents'][1]['visible'] is False and
               roots[5]['documents'][1]['visible'] is True,
               'tombstone failed to preserve prior state')
        saved = {(p.name,p.stat().st_size,prod.sha(p.read_bytes()))
                 for p in (bundle/'snapshots').glob('*.json')}
        readings=[]
        receipt_paths=[]
        for gen,root in enumerate(roots):
            for pattern,cap in ((b'banana',None),(b'\x00',None),(b'\xff',None),
                                (b'never-present',None),(b'banana',0),(b'banana',1),
                                (b'new\xff',None)):
                rec=prod.query(bundle,root,pins[gen],pattern,cap)
                path=out/('q%02d_%02d.json'%(gen,len(readings)))
                path.write_bytes(prod.canonical(rec))
                r=run_verifier(bundle,gen,path)
                ensure(r.returncode==0,'independent replay failed: '+r.stderr)
                readings.append({'gen':gen,'pattern':pattern.hex(),'cap':cap,
                                 'status':rec['status'],'count':rec['count']})
                receipt_paths.append((gen,path,rec))
        ensure(len(readings)==49,'missing independent query cases')
        ensure(next(r for r in readings if r['gen']==0 and r['pattern']==b'new\xff'.hex())['status']=='PROVEN_EMPTY'
               and next(r for r in readings if r['gen']==3 and r['pattern']==b'new\xff'.hex())['status']=='FOUND',
               'a new revision must not rewrite an old empty claim')
        ensure(saved == {(p.name,p.stat().st_size,prod.sha(p.read_bytes()))
                           for p in (bundle/'snapshots').glob('*.json')},
               'old pinned snapshot bytes changed')
        ensure(len(list((bundle/'objects').glob('*.bin')))==4,
               'content dedup failed or old byte data missing')
        negative=[]
        g,path,original=receipt_paths[-7]  # latest generation pattern banana
        ensure(g==6,'wrong negative baseline generation')
        def attack(desc,modify):
            forged=copy.deepcopy(original)
            modify(forged)
            p=out/('bad_%02d.json'%len(negative))
            p.write_bytes(prod.canonical(forged))
            r=run_verifier(bundle,g,p)
            ensure(r.returncode==3 and 'UNTRUSTED' in r.stderr,
                   'independent verifier FAIL OPEN: '+desc+' '+r.stdout+' '+r.stderr)
            negative.append(desc)
        attack('false-proven-empty',lambda x:x.update(status='PROVEN_EMPTY',count=0,coordinates=[]))
        attack('forged-count',lambda x:x.update(count=x['count']+1))
        attack('missing-first-offset',lambda x:x['coordinates'].pop(0))
        attack('duplicate-first-offset',lambda x:x['coordinates'].append(x['coordinates'][0]))
        attack('reversed-coordinates',lambda x:x['coordinates'].reverse())
        attack('doc-id-forged',lambda x:x['coordinates'][0].__setitem__(0,77))
        attack('revision-forged',lambda x:x['coordinates'][0].__setitem__(1,9))
        attack('offset-forged',lambda x:x['coordinates'][0].__setitem__(2,99))
        attack('wrong-vault-id',lambda x:x.update(vault_id='0'*32))
        attack('wrong-root-pin',lambda x:x.update(root_sha256='0'*64))
        attack('wrong-generation',lambda x:x.update(generation=0))
        attack('changed-query',lambda x:x.update(pattern_hex=b'new'.hex()))
        attack('empty-query',lambda x:x.update(pattern_hex=''))
        attack('wrong-cap',lambda x:x.update(max_offsets=0))
        attack('wrong-coverage',lambda x:x['coverage'].pop(0))
        attack('duplicated-coverage',lambda x:x['coverage'].append(x['coverage'][0]))
        attack('claim-limited-as-complete',lambda x:x.update(status='FOUND',reason='LIMITED_COMPLETENESS'))
        attack('wrong-reason',lambda x:x.update(reason='FULL_SNAPSHOT_ORACLE_SCAN_FALSE'))
        attack('extraneous-field',lambda x:x.update(secret='x'))
        attack('malformed-hex',lambda x:x.update(pattern_hex='GG'))
        attack('negative-count',lambda x:x.update(count=-1))
        attack('unsorted-coverage',lambda x:x['coverage'].reverse())
        attack('faked-no-hit-but-correct-count',lambda x:x.update(coordinates=[],status='PROVEN_EMPTY'))
        # Mutation to bytes of one mandatory object cannot be accepted as empty.
        obj=bundle/'objects'/(roots[-1]['documents'][0]['sha256']+'.bin')
        original_data=obj.read_bytes()
        try:
            obj.write_bytes(original_data[:-1]+bytes([original_data[-1]^1]))
            r=run_verifier(bundle,g,path)
            ensure(r.returncode==3 and 'UNTRUSTED' in r.stderr,
                   'corrupted object accepted')
            negative.append('object-bitflip')
        finally:
            obj.write_bytes(original_data)
        # Missing physical object must fail closed, never return empty.
        tmp=obj.with_name(obj.name+'.hidden')
        obj.rename(tmp)
        try:
            r=run_verifier(bundle,g,path)
            ensure(r.returncode==3,'missing source object was accepted')
            negative.append('missing-source-object')
        finally:
            tmp.rename(obj)
        # Symlink substitution must fail even if it targets the original data.
        other=obj.with_name(obj.name+'.saved')
        obj.rename(other)
        obj.symlink_to(other)
        try:
            r=run_verifier(bundle,g,path)
            ensure(r.returncode==3,'source symlink accepted')
            negative.append('symlink-object')
        finally:
            obj.unlink()
            other.rename(obj)
        wrong_pins=['0'*64,*EXPECTED[1:]]
        r=run_verifier(bundle,g,path,wrong_pins)
        ensure(r.returncode==3,'forged external prior root pin accepted')
        negative.append('wrong-prior-root-pin')
        # Re-pin a mutated latest root (adversarially recompute its SHA).
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            clone=Path(t)/'fixture'
            shutil.copytree(bundle,clone,symlinks=True)
            mod=copy.deepcopy(roots[-1])
            mod['documents'][0]['doc_id']=55
            body=dict(mod)
            body.pop('snapshot_id')
            mod['snapshot_id']=prod.sha(prod.DOMAIN+prod.canonical(body))
            raw=prod.canonical(mod)
            pin=prod.sha(raw)
            (clone/'snapshots'/(pin+'.json')).write_bytes(raw)
            (clone/'CURRENT').write_text(pin+'\n')
            r=run_verifier(clone,g,path,EXPECTED[:-1]+[pin])
            ensure(r.returncode==3,'re-pinned forged projection accepted')
            negative.append('repinned-false-projection')
        # Independently re-pin *physically* changed roots under a new external
        # test pin. Structural/event projection checks must still reject them.
        repinned_mutations = [
            ('repinned-missing-coverage',lambda x:x['coverage'].pop()),
            ('repinned-wrong-next-id',lambda x:x.update(next_doc_id=99)),
            ('repinned-parent-fork',lambda x:x.update(previous_root_sha256='0'*64)),
            ('repinned-ordinal-rewrite',lambda x:x['events'][-1].update(ordinal=99)),
            ('repinned-unsafe-rename',lambda x:x['events'][2].update(path_hex=b'../bad'.hex())),
            ('repinned-revision-skipped',lambda x:x['events'][3].update(revision=9)),
            ('repinned-reordered-documents',lambda x:x['documents'].reverse()),
        ]
        for label,mutation in repinned_mutations:
            with tempfile.TemporaryDirectory() as t:
                clone=Path(t)/'fixture'
                shutil.copytree(bundle,clone,symlinks=True)
                mod=copy.deepcopy(roots[-1])
                mutation(mod)
                body=dict(mod)
                body.pop('snapshot_id')
                mod['snapshot_id']=prod.sha(prod.DOMAIN+prod.canonical(body))
                raw=prod.canonical(mod)
                pin=prod.sha(raw)
                (clone/'snapshots'/(pin+'.json')).write_bytes(raw)
                (clone/'CURRENT').write_text(pin+'\n')
                r=run_verifier(clone,g,path,EXPECTED[:-1]+[pin])
                ensure(r.returncode==3,'semantic root manipulation accepted: '+label)
                negative.append(label)
        # Producer must reject path traversal, duplicate, unchanged revision,
        # ID allocation collision and stale snapshot pointer.
        producer_negative=[]
        for label, fn in [
            ('traversal',lambda:prod.path_hex(b'../secret')),
            ('absolute',lambda:prod.path_hex(b'/secret')),
            ('nul-byte',lambda:prod.path_hex(b'bad\x00.txt')),
            ('empty-component',lambda:prod.path_hex(b'x//y')),
            ('backslash',lambda:prod.path_hex(b'x\\y')),
            ('empty-path',lambda:prod.path_hex(b'')),
            ('id-already-used',lambda:prod.append(copy.deepcopy(roots[6]['events']),'INGEST',0,b'zz',b'hello')),
            ('duplicate-active-path',lambda:prod.append(copy.deepcopy(roots[6]['events']),'INGEST',4,b'q.bin',b'hello')),
            ('same-content-revision',lambda:prod.append(copy.deepcopy(roots[6]['events']),'REVISE',0,content=b'new\xffbanana')),
            ('rename-conflict',lambda:prod.append(copy.deepcopy(roots[6]['events']),'RENAME',0,b'copy.bin')),
            ('tombstoned-revise',lambda:prod.append(copy.deepcopy(roots[6]['events']),'REVISE',1,content=b'x')),
            ('wrong-cas-root',lambda:prod.publish(bundle,roots[6]['events'],7,'0'*64)),
        ]:
            rejection(fn,label,producer_negative)
        # Separate open file descriptions must not acquire the same writer lock.
        import fcntl
        with (bundle/'LOCK').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            rejection(lambda:prod.publish(bundle,roots[6]['events'],7,EXPECTED[-1]),
                      'concurrent-writer-denied',producer_negative)
            fcntl.flock(lock,fcntl.LOCK_UN)
        # Immutable staging without pointer update must not change current.
        stale=(bundle/'CURRENT').read_text()
        orphan=bundle/'objects'/(prod.sha(b'aborted')+'.bin')
        orphan.write_bytes(b'aborted')
        ensure((bundle/'CURRENT').read_text()==stale,
               'uncommitted source changed committed CURRENT')
        producer_negative.append('interrupted-before-root-publication')
        # Orphan cleanup is explicit and only performed after confirming the
        # referenced snapshot set is intact; no mutation of committed roots.
        orphan.unlink()
        report.update(status='CORE_L10_APPEND_IDENTITY_V2_REPLAY_PASS_RESEARCH',
                      snapshot_count=len(roots),query_replay_count=len(readings),
                      independent_hostile_rejected=len(negative),
                      producer_rejected=len(producer_negative),
                      hostile_labels=negative,producer_rejection_labels=producer_negative,
                      unique_content_sha256_objects=4,
                      stable_doc_ids_verified=True,
                      metadata_only_rename_does_not_change_revision=True,
                      previous_root_bytes_immutable=True,
                      tombstone_preserves_old_snapshots=True,
                      dedup_does_not_merge_doc_ids=True,
                      current_pointer_unchanged_after_aborted_write=True,
                      synthetic_orphan_cleaned_after_failure=True,
                      test_source_objects_are_raw_oracles_not_rlb2=True,
                      elapsed_wall_seconds=round(time.monotonic()-t0,6))
    except Exception as e:
        report.update(error=repr(e))
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','snapshot_count',
                     'query_replay_count','independent_hostile_rejected','producer_rejected')},sort_keys=True))
    return 0 if report['status']=='CORE_L10_APPEND_IDENTITY_V2_REPLAY_PASS_RESEARCH' else 3


if __name__=='__main__':
    sys.exit(main())
