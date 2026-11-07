#!/usr/bin/env python3
"""Fixture/append/replay driver; all generated data stays in disposable output."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOC = REPO/'experiments/loc_text_ef_v0'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(path):
    return {str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
            for p in sorted(path.rglob('*')) if p.is_file()}


def run(args, log, expect=0, timeout=600):
    start = time.monotonic()
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    log.write_text(result.stdout+result.stderr)
    assert result.returncode == expect, (list(map(str, args)), result.returncode, result.stderr[-1200:])
    return result, dict(command=list(map(str, args)), exit_code=result.returncode, wall_seconds=time.monotonic()-start)


def queries():
    patterns = [b'shared', b'dup', b'\0\xff', b'\xff\0', b'\0', b'\xff', b'alpha LEFT', b'a',
        b'not-present', b'ONLY_IN_V1', b'SHARED', b'shared middle', b'shared  middle',
        b'\0\xffshared', b'dupdup', b'LEFTEDGE-A', b'x'*128]
    result = [dict(name='q'+str(i), hex=p.hex(), limit=256) for i, p in enumerate(patterns)]
    result += [dict(name='cap'+str(i), hex=b'shared'.hex(), limit=i) for i in (0, 1, 2)]
    return result


def materialize(bundle, folder, docs, logs):
    for i, doc in enumerate(docs):
        out = folder/(str(i)+'.raw')
        if out.exists():
            assert len(out.read_bytes()) == doc['byte_length'] and sha(out.read_bytes()) == doc['sha256']
            continue
        if doc['byte_length'] == 0:
            out.write_bytes(b'')
        else:
            leaf = bundle/'leaves'/str(i)
            args = [sys.executable, LOC/'trusted_materialize_v0.py', '--manifest', leaf/'root.json',
                '--root-sha256', sha((leaf/'root.json').read_bytes()), '--rlb2', leaf/'rlb2', '--rlr2', leaf/'rlr2',
                '--ef', leaf/'ef', '--decoder', bundle/'decoder', '--output', out, '--threads', '2', '--mode', 'rank']
            result, _ = run(args, logs/('materialize-'+str(i)+'.log'))
            assert json.loads(result.stdout)['verified_source_before_publication']
        assert len(out.read_bytes()) == doc['byte_length'] and sha(out.read_bytes()) == doc['sha256']


def verifier_args(bundle, root, reference, oracles, receipt, query, copied):
    data = json.loads(root.read_bytes())
    args = [sys.executable, '-I', copied, '--bundle', bundle, '--root', root,
        '--pin', sha(root.read_bytes()), '--generation', str(data['generation']), '--oracles', oracles,
        '--receipt', receipt, '--query-hex', query['hex'], '--limit', str(query['limit']),
        '--research', REPO, '--reference', reference]
    if data['generation']:
        old = root.with_name(root.stem[0]+'0.json')
        args += ['--previous-root', old, '--previous-pin', sha(old.read_bytes())]
    return list(map(str, args))


def replay(bundle, reference, oracles, receipts, output, cases):
    output.mkdir()
    copied = output/'standalone.py'
    shutil.copy2(HERE/'verifier.py', copied)
    imports = []
    for node in ast.walk(ast.parse(copied.read_text())):
        if isinstance(node, ast.Import):
            imports += [x.name.split('.')[0] for x in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.append(node.module.split('.')[0])
    assert set(imports) <= sys.stdlib_module_names and 'producer' not in imports
    results = []
    for name in ('A0', 'B0', 'A1', 'B1'):
        root = bundle/'roots'/(name+'.json')
        if not root.exists():
            continue
        docs = json.loads(root.read_bytes())['documents']
        for case in cases:
            pattern = bytes.fromhex(case['hex'])
            oracle = []
            for i, d in enumerate(docs):
                raw = (oracles/(str(i)+'.raw')).read_bytes()
                assert sha(raw) == d['sha256']
                oracle += [[i, j] for j in range(len(raw)) if raw.startswith(pattern, j)]
            path = receipts/name/(case['name']+'.json')
            r = json.loads(path.read_bytes())
            assert r['match_count'] == len(oracle) and r['coordinates'] == oracle[:case['limit']]
            expected_exit = 0 if len(oracle) <= case['limit'] else 3
            p, command = run(verifier_args(bundle, root, reference, oracles, path, case, copied),
                output/(name+'-'+case['name']+'.log'), expect=expected_exit)
            info = json.loads(p.stdout if expected_exit == 0 else p.stderr)
            assert info['count_verified'] and info['full_leaf_block_coverage']
            assert info['status'] == ('VERIFIED_APPEND_RESEARCH' if expected_exit == 0 else 'UNTRUSTED')
            results.append(dict(snapshot=name, case=case, count=len(oracle), coordinates=r['coordinates'],
                                status=r['status'], exit_code=expected_exit, verification=info, command=command))
    # Equal semantic answers across partitions, without aliasing physical roots.
    for generation in (0, 1):
        a = {r['case']['name']: r for r in results if r['snapshot'] == 'A'+str(generation)}
        b = {r['case']['name']: r for r in results if r['snapshot'] == 'B'+str(generation)}
        for key in a:
            assert (a[key]['count'], a[key]['coordinates'], a[key]['status']) == (b[key]['count'], b[key]['coordinates'], b[key]['status'])
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('produce', 'independent'), required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--capsule', type=Path)
    p.add_argument('--expected-roots-json')
    a = p.parse_args()
    target, reference = a.output.resolve(), a.reference.resolve()
    assert not target.exists()
    target.mkdir()
    logs = target/'logs'
    logs.mkdir()
    oracles = target/'ephemeral-oracles'
    oracles.mkdir()
    commands, results = [], []
    try:
        if a.phase == 'produce':
            assert a.decoder is not None
            bundle, receipts = target/'bundle', target/'receipts'
            receipts.mkdir()
            _, cmd = run([sys.executable, '-I', HERE/'fixture.py', '--stage', 'initial', '--bundle', bundle,
                          '--reference', reference, '--decoder', a.decoder], logs/'construct-v0.log')
            commands.append(cmd)
            cases = queries()
            (target/'queries.json').write_text(json.dumps(cases, sort_keys=True)+'\n')
            old = inventory(bundle)
            trusted_pins = {n:sha((bundle/'roots'/(n+'.json')).read_bytes()) for n in ('A0','B0')}
            for generation in (0, 1):
                if generation:
                    _, cmd = run([sys.executable, '-I', HERE/'fixture.py', '--stage', 'append', '--bundle', bundle,
                                  '--reference', reference], logs/'construct-v1.log')
                    commands.append(cmd)
                    after = inventory(bundle)
                    assert all(after.get(k) == v for k, v in old.items()), 'old bytes rewritten by append'
                    for n in ('A0','B0'):
                        assert sha((bundle/'roots'/(n+'.json')).read_bytes()) == trusted_pins[n]
                    trusted_pins.update({n:sha((bundle/'roots'/(n+'.json')).read_bytes()) for n in ('A1','B1')})
                for layout in ('A', 'B'):
                    name = layout+str(generation)
                    root = bundle/'roots'/(name+'.json')
                    data = json.loads(root.read_bytes())
                    materialize(bundle, oracles, data['documents'], logs)
                    withheld = target/'withheld'
                    oracles.rename(withheld)
                    try:
                        args = [sys.executable, '-I', HERE/'producer.py', '--bundle', bundle, '--root', root,
                            '--pin', sha(root.read_bytes()), '--generation', str(generation), '--reference', reference,
                            '--queries', target/'queries.json', '--output', receipts/name]
                        if generation:
                            parent = bundle/'roots'/(layout+'0.json')
                            args += ['--previous-root', parent, '--previous-pin', sha(parent.read_bytes())]
                        _, cmd = run(args, logs/('produce-'+name+'.log'))
                        commands.append(cmd)
                    finally:
                        withheld.rename(oracles)
                if generation == 0:
                    results += replay(bundle, reference, oracles, receipts, target/'preappend-replay', cases)
                    old_receipts = inventory(receipts)
            # Reuse the ORIGINAL V0 receipts after append; never regenerate them.
            now_receipts = inventory(receipts)
            assert all(now_receipts.get(k) == v for k,v in old_receipts.items()), 'old V0 receipt bytes changed'
            (target/'preservation.json').write_text(json.dumps(dict(old_files=old, all_old_bytes_unchanged=True,
                old_receipts=old_receipts, old_receipt_bytes=sum(v['bytes'] for v in old_receipts.values()),
                old_file_bytes=sum(v['bytes'] for v in old.values()), new_files={k:v for k,v in after.items() if k not in old}),
                sort_keys=True, indent=2)+'\n')
            (target/'trusted-pins.json').write_text(json.dumps(trusted_pins,sort_keys=True)+'\n')
        else:
            assert a.capsule is not None and a.expected_roots_json is not None
            expected_pins = json.loads(a.expected_roots_json)
            assert set(expected_pins) == {'A0','B0','A1','B1'}
            for n,pin in expected_pins.items():
                assert sha((a.capsule/'bundle/roots'/(n+'.json')).read_bytes()) == pin, 'externally expected snapshot pin mismatch'
            inv = json.loads((a.capsule/'inventory.json').read_bytes())
            observed = inventory(a.capsule)
            observed.pop('inventory.json')
            assert observed == inv and sum(x['bytes'] for x in observed.values()) <= 4*1024**2
            assert not any(p.is_symlink() for p in a.capsule.rglob('*'))
            bundle, receipts = target/'bundle', target/'receipts'
            shutil.copytree(a.capsule/'bundle', bundle)
            (bundle/'decoder').chmod(0o700)
            shutil.copytree(a.capsule/'receipts', receipts)
            shutil.copy2(a.capsule/'queries.json', target/'queries.json')
            shutil.copy2(a.capsule/'preservation.json', target/'preservation.json')
            cases = json.loads((target/'queries.json').read_bytes())
            materialize(bundle, oracles, json.loads((bundle/'roots/A1.json').read_bytes())['documents'], logs)
        results += replay(bundle, reference, oracles, receipts, target/'postappend-replay', cases)
        _, cmd = run([sys.executable, '-I', HERE/'hostiles.py', '--bundle', bundle, '--receipts', receipts,
                      '--oracles', oracles, '--reference', reference, '--output', target/'hostiles'], logs/'hostiles.log')
        commands.append(cmd)
        negatives = json.loads((target/'hostiles/result.json').read_bytes())
        assert negatives['hostile_rejections'] >= 25
        roots = {n:json.loads((bundle/'roots'/(n+'.json')).read_bytes()) for n in ('A0','B0','A1','B1')}
        for gen in (0,1):
            x,y = roots['A'+str(gen)], roots['B'+str(gen)]
            assert x['runtime_corpus_id'] == y['runtime_corpus_id'] and x['source_manifest_id'] == y['source_manifest_id']
            assert x['research_composition_root_id'] != y['research_composition_root_id']
        initial = json.loads((receipts/'A0/q9.json').read_bytes())
        appended = json.loads((receipts/'A1/q9.json').read_bytes())
        assert initial['status'] == 'PROVEN_EMPTY' and appended['status'] == 'FOUND' and appended['coordinates'] == [[9,0]]
        assert json.loads((receipts/'A0/q1.json').read_bytes())['coordinates'] == [[5,0],[6,0]]
        assert json.loads((receipts/'A1/q1.json').read_bytes())['coordinates'] == [[5,0],[6,0],[10,0]]
        if a.phase == 'produce':
            capsule = target/'capsule'
            capsule.mkdir()
            shutil.copytree(bundle, capsule/'bundle')
            shutil.copytree(receipts, capsule/'receipts')
            for name in ('queries.json','preservation.json'):
                shutil.copy2(target/name, capsule/name)
            assert sum(v['bytes'] for v in inventory(capsule).values()) <= 4*1024**2
            (capsule/'inventory.json').write_text(json.dumps(inventory(capsule),sort_keys=True,indent=2)+'\n')
        costs = inventory(bundle)
        preservation = json.loads((target/'preservation.json').read_bytes())
        # Inventory external support and private snapshots separately. These
        # are not free, and file-length sums are not peak allocated disk/RAM.
        support_paths = list(HERE.glob('*.*')) + [
            LOC/'real_enwik8_ef_physical_gate.py', LOC/'native_run_stream_inverse_rlb2_ef.cpp',
            REPO/'.github/workflows/core-l10-incremental-compose.yml']
        support = {str(f.relative_to(REPO)): dict(bytes=f.stat().st_size, sha256=sha(f.read_bytes()))
                   for f in support_paths if f.is_file()}
        snapshot_bytes = {}
        for n, root in roots.items():
            amounts = []
            for i, doc in enumerate(root['documents']):
                if doc['byte_length']:
                    binding = json.loads((bundle/'leaves'/str(i)/'binding.json').read_bytes())
                    amounts.append(root['dependencies']['decoder']['bytes']+
                                   sum(v['bytes'] for v in binding['artifacts'].values()))
            snapshot_bytes[n] = dict(producer_private_component_decoder_copies=sum(amounts),
                materializer_one_leaf_private_component_decoder_copy_max=max(amounts),
                materializer_one_candidate_source_max=max(d['byte_length'] for d in root['documents']))
        git_stores = set()
        for checkout in (REPO, reference):
            common = Path(subprocess.check_output(['git','-C',str(checkout),'rev-parse','--git-common-dir'],text=True).strip())
            git_stores.add((common if common.is_absolute() else checkout/common).resolve())
        report = dict(status='VERIFIED_APPEND_RESEARCH', head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            roots={k:dict(root_sha256=sha((bundle/'roots'/(k+'.json')).read_bytes()), research_id=v['research_composition_root_id'],
                         runtime_corpus_id=v['runtime_corpus_id'], source_manifest_id=v['source_manifest_id'], documents=len(v['documents']), blocks=len(v['blocks'])) for k,v in roots.items()},
            actual_leaf_profiles='real frozen RankV2/RLB2/RLR2/EF; manifest-only zero document',
            query_replays=len(results), per_snapshot_cases=len(cases), hostiles=negatives, verification_results=results,
            preserved_v0_file_count=len(preservation['old_files']), preserved_v0_file_bytes=preservation['old_file_bytes'],
            preserved_v0_receipt_count=len(preservation['old_receipts']), preserved_v0_receipt_bytes=preservation['old_receipt_bytes'],
            new_append_file_count=len(preservation['new_files']), new_append_file_bytes=sum(v['bytes'] for v in preservation['new_files'].values()),
            bundle_file_bytes=sum(v['bytes'] for v in costs.values()), bundle_identities=costs,
            receipt_bytes=sum(f.stat().st_size for f in receipts.rglob('*.json')),
            code_bytes=sum(f.stat().st_size for f in HERE.glob('*.py')),
            runtime_dependency_code_bytes=sum(v['bytes'] for v in roots['A0']['dependencies']['code'].values()),
            support_code_and_docs_identities=support, snapshot_file_bytes=snapshot_bytes,
            research_reference_checkout_file_bytes=sum(f.stat().st_size for p in (REPO, reference)
                for f in p.rglob('*') if f.is_file() and '.git' not in f.relative_to(p).parts),
            git_store_file_bytes=sum(f.stat().st_size for p in map(Path,git_stores) for f in p.rglob('*') if f.is_file()),
            ephemeral_source_bytes=sum(f.stat().st_size for f in oracles.iterdir()), retained_source_archive_bytes=0,
            scratch_file_bytes=sum(f.stat().st_size for f in target.rglob('*') if f.is_file()),
            producer_metrics={n:json.loads((logs/('produce-'+n+'.log')).read_text()) for n in roots} if a.phase=='produce' else None,
            limits=['bounded fixture: <=64 documents, <=256B/document, <=256 returned coordinates',
                    'full all-block/all-leaf fanout; source scan plus bounded per-leaf quadratic physical oracle',
                    'unsigned expected roots; no RFC9162 Merkle consistency, signatures/currentness or fork prevention',
                    'non-normative physical bindings, not historical operator blocks or Index Forest',
                    'no multi-TB, ABI, production, SLA or full installed footprint claim'])
        (target/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        print(json.dumps(report,sort_keys=True),flush=True)
        print('CORE_L10_VERIFIED_APPEND_RESEARCH_PASS',flush=True)
    finally:
        shutil.rmtree(oracles)
        (target/'commands.json').write_text(json.dumps(commands,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':
    main()
