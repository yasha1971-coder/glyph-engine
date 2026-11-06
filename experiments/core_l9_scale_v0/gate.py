#!/usr/bin/env python3
"""CORE-L9-S construction/measurement orchestration. No implementation imports.

Source exists only in an ephemeral oracle directory and is removed on exit.
Search runs in its own process with only query bytes + physical state paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOC = REPO/'experiments/loc_text_ef_v0'
REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
SOURCE = '2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def run(args, output, timeout=2400):
    started = time.monotonic()
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    output.write_text(result.stdout+result.stderr)
    assert result.returncode == 0, (result.returncode, str(output), result.stderr[-1000:])
    return dict(command=list(map(str, args)), exit_code=result.returncode, wall_seconds=time.monotonic()-started)


def choose_queries(oracle, output):
    assert oracle.stat().st_size == 100000000 and digest(oracle) == SOURCE
    terms = [
        b'United Nations Security Council', b'United Nations Universal Declaration of Human Rights',
        b'Constitution of the United States of America', b'International Committee of the Red Cross',
        b'National Aeronautics and Space Administration', b'University of California, Berkeley',
        b'United States Declaration of Independence', b'Massachusetts Institute of Technology',
        b'International Monetary Fund', b'Universal Declaration of Human Rights',
        b'the University of Cambridge', b'the University of Oxford', b'Articles of Confederation',
        b'the Republic of South Africa', b'the Holy Roman Empire', b'the Treaty of Versailles',
        b'American Civil Liberties Union', b'International Court of Justice',
        b'Organisation for Economic Co-operation and Development',
        b'European Coal and Steel Community', b'the University of Chicago',
        b'United States House of Representatives', b'the Library of Congress',
        b'American Declaration of Independence', b'the University of Edinburgh']
    with oracle.open('rb') as f:
        for pos in (1200, 1000000, 13579113, 27890000, 55432100, 83567543):
            f.seek(pos)
            terms.append(f.read(48))
        f.seek(99999937)
        near_end = f.read(48)
        f.seek(1024**2-7)
        boundary = f.read(32)
        nonascii = None
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            match = re.search(rb'[\x80-\xff]', chunk)
            if match:
                start = max(0, match.start()-8)
                nonascii = chunk[start:start+48]
                break
    assert nonascii and any(b >= 128 for b in nonascii), 'no non-ASCII query found'
    # One full bounded-buffer test-oracle pass chooses rare repeated phrases.
    # It is never passed to producer count/locate/excerpt code.
    counts = [0]*len(terms)
    with oracle.open('rb') as f:
        carries = [b'']*len(terms)
        for chunk in iter(lambda: f.read(1024**2), b''):
            for i, pattern in enumerate(terms):
                window = carries[i]+chunk
                at = 0
                while True:
                    at = window.find(pattern, at)
                    if at < 0:
                        break
                    counts[i] += 1
                    at += 1
                carries[i] = window[-(len(pattern)-1):]
    selected = sorted((c, i) for i, c in enumerate(counts) if 2 <= c <= 16)[:2]
    assert len(selected) == 2, 'two rare multi-hit phrases not found; do not fake counts'
    result = [dict(name='multi-'+str(k+1), pattern_hex=terms[i].hex(), limit=256)
              for k, (_, i) in enumerate(selected)]
    result += [dict(name=name, pattern_hex=pattern.hex(), limit=256) for name, pattern in (
        ('verified-empty', b'\x00GLYPH_CORE_L9S_ABSENT\x00'), ('near-end', near_end),
        ('non-ascii', nonascii), ('chunk-boundary', boundary), ('limited-common', b'e'))]
    (output/'selection.json').write_text(json.dumps(dict(candidate_counts=[
        dict(pattern_hex=p.hex(), count=c) for p, c in zip(terms, counts)],
        selected_counts=[c for c, _ in selected], near_end_source_offset=99999937,
        chunk_boundary_source_offset=1024**2-7,
        full_selection_pass_bytes=100000000, initial_source_hash_bytes=100000000,
        note='test-oracle pattern selection; excluded from producer search input'), sort_keys=True, indent=2)+'\n')
    return result


def materialize(state, output, log):
    return run([sys.executable, LOC/'trusted_materialize_v0.py', '--manifest', state/'root.json',
        '--root-sha256', digest(state/'root.json'), '--rlb2', state/'rlb2', '--rlr2', state/'rlr2',
        '--ef', state/'ef', '--decoder', state/'decoder', '--output', output,
        '--threads', '4', '--mode', 'rank'], log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('small', 'produce', 'replay'), required=True)
    for name in ('reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--capsule', type=Path)
    p.add_argument('--expected-root')
    a = p.parse_args()
    target, ref = a.output.resolve(), a.reference.resolve()
    assert not target.exists(), 'new disposable output required'
    assert subprocess.check_output(['git', '-C', str(ref), 'rev-parse', 'HEAD'], text=True).strip() == REF
    target.mkdir(parents=True)
    commands = []
    oracle = target/'ephemeral-oracle'
    try:
        if a.phase == 'replay':
            assert a.capsule is not None and a.expected_root is not None
            assert digest(a.capsule/'state/root.json') == a.expected_root, 'external job-output root differs'
            inventory = json.loads((a.capsule/'inventory.json').read_bytes())
            paths = [f for f in a.capsule.rglob('*') if f.is_file()]
            assert not any(f.is_symlink() for f in a.capsule.rglob('*'))
            assert sum(f.stat().st_size for f in paths) <= 96*1024**2
            actual = {str(f.relative_to(a.capsule)) for f in paths if f.name != 'inventory.json'}
            assert actual == set(inventory['files'])
            for name, record in inventory['files'].items():
                f = a.capsule/name
                assert f.stat().st_size == record['bytes'] and digest(f) == record['sha256']
            state = target/'state'
            shutil.copytree(a.capsule/'state', state)
            (state/'decoder').chmod(0o700)
            shutil.copy2(a.capsule/'queries.json', target/'queries.json')
            shutil.copytree(a.capsule/'receipts', target/'receipts')
            commands.append(materialize(state, oracle, target/'materialize.log'))
        elif a.phase == 'small':
            assert a.decoder is not None
            fixture = target/'fixture'
            commands.append(run([sys.executable, '-I', REPO/'experiments/core_l9_exact_query_v0/fixture.py',
                '--reference', ref, '--decoder', a.decoder, '--output', fixture], target/'construction.log'))
            state = target/'state'
            shutil.move(fixture/'state', state)
            shutil.move(fixture/'oracle/materialized', oracle)
            shutil.rmtree(fixture)
            queries = [dict(name=name, pattern_hex=pattern.hex(), limit=limit) for name, pattern, limit in (
                ('multi-1', b'banana', 256), ('multi-2', b'ana', 256), ('verified-empty', b'absent', 256),
                ('binary-nul', b'\x00', 256), ('binary-ff', b'\xff', 256), ('limited-common', b'a', 1))]
            (target/'queries.json').write_text(json.dumps(queries, sort_keys=True)+'\n')
        else:
            assert a.decoder is not None
            l8 = target/'l8-rebuild'
            commands.append(run([sys.executable, LOC/'core_l8_enwik8_pinned_materialize_gate.py',
                '--reference', ref, '--decoder', a.decoder, '--verifier', LOC/'trusted_materialize_v0.py',
                '--replay', LOC/'independent_materialization_replay_v0.py', '--output', l8],
                target/'l8-rebuild.log', timeout=2800))
            result = json.loads((l8/'result.json').read_bytes())
            assert result['status'] == 'CORE_L8_PINNED_ENWIK8_MATERIALIZATION_RESEARCH_PASS'
            assert result['canonical_source_bytes'] == 100000000 and result['canonical_source_sha256'] == SOURCE
            assert result['retained_index_bytes'] == 89053457
            assert result['independent_offline_byte_replay']['status'] == 'CORE_L8_OFFLINE_SOURCE_IDENTITY_REPLAY_PASS'
            assert result['no_source_sa_bwt_files_used_in_restore']
            state = target/'state'
            state.mkdir()
            for key, name in [('rlb2', 'retained.rlb2'), ('rlr2', 'retained.rlr2'), ('ef', 'retained.loc-ef'),
                              ('root.json', 'l8-manifest.json')]:
                shutil.move(l8/'work'/name, state/key)
            shutil.copy2(a.decoder, state/'decoder')
            shutil.move(l8/'work/final-verified-source-enwik8', oracle)
            shutil.copy2(l8/'result.json', target/'prerequisite.json')
            shutil.copy2(l8/'commands.json', target/'prerequisite-commands.json')
            shutil.rmtree(l8)
            queries = choose_queries(oracle, target)
            (target/'queries.json').write_text(json.dumps(queries, sort_keys=True)+'\n')
        assert set(x.name for x in state.iterdir()) == {'rlb2', 'rlr2', 'ef', 'root.json', 'decoder'}
        if a.phase != 'replay':
            # Withhold the oracle entirely while the producer process runs.
            # It receives only physical state + bounded exact query bytes.
            withheld = target/'withheld-oracle'
            oracle.rename(withheld)
            try:
                command = [sys.executable, '-I', HERE/'produce_batch.py', '--state', state, '--reference', ref,
                           '--queries', target/'queries.json', '--output', target/'receipts']
                if a.phase == 'small':
                    command.append('--test-fixture')
                commands.append(run(command, target/'producer.log', timeout=1800))
            finally:
                withheld.rename(oracle)
        command = [sys.executable, '-I', HERE/'replay_gate.py', '--reference', ref, '--state', state,
                   '--oracle', oracle, '--queries', target/'queries.json', '--receipts', target/'receipts',
                   '--output', target/'replay']
        if a.phase == 'small':
            command.append('--test-fixture')
        commands.append(run(command, target/'replay.log', timeout=1800))
        replay = json.loads((target/'replay/result.json').read_bytes())
        assert replay['status'] == ('VERIFIED_SMALL_SCAN' if a.phase == 'small' else 'VERIFIED_100MB_SCAN')
        if a.phase != 'small':
            queries = json.loads((target/'queries.json').read_bytes())
            cases = replay['queries']
            byname = {c['query']['name']: c for c in cases}
            assert len(cases) >= 6 and byname['verified-empty']['verifier']['claim'] == 'PROVEN_EMPTY'
            assert all(byname[n]['verifier']['independent_count'] >= 2 for n in ('multi-1', 'multi-2'))
            assert byname['limited-common']['verifier']['independent_count'] > 256
            assert byname['limited-common']['verifier']['status'] == 'UNTRUSTED'
            if a.phase == 'produce':
                capsule = target/'capsule'
                capsule.mkdir()
                shutil.copytree(state, capsule/'state')
                shutil.copytree(target/'receipts', capsule/'receipts')
                shutil.copy2(target/'queries.json', capsule/'queries.json')
                size = sum(f.stat().st_size for f in capsule.rglob('*') if f.is_file())
                assert size <= 96*1024**2, 'bounded index-only upload cap'
                (capsule/'inventory.json').write_text(json.dumps(dict(payload_bytes=size, upload_cap=96*1024**2,
                    files={str(f.relative_to(capsule)): dict(bytes=f.stat().st_size, sha256=digest(f))
                           for f in capsule.rglob('*') if f.is_file()}, source_archive_bytes=0), sort_keys=True, indent=2)+'\n')
        root = json.loads((state/'root.json').read_bytes())
        retained = {key: dict(bytes=f.stat().st_size, sha256=digest(f)) for key, f in [
            ('rlb2', state/'rlb2'), ('rlr2', state/'rlr2'), ('ef', state/'ef'),
            ('manifest', state/'root.json'), ('decoder', state/'decoder')]}
        report = dict(status=replay['status'], head=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
            frozen_commit=REF, source=root['source'], root_sha256=digest(state/'root.json'), retained=retained,
            retained_index_bytes=sum(retained[k]['bytes'] for k in ('rlb2', 'rlr2', 'ef')),
            retained_receipt_bytes=sum(f.stat().st_size for f in (target/'receipts').glob('*.json') if f.name != 'producer-result.json'),
            retained_producer_metadata_bytes=(target/'receipts/producer-result.json').stat().st_size,
            retained_raw_source_archive_bytes=0, ephemeral_oracle_bytes=root['source']['bytes'],
            measured_scratch_file_bytes=sum(f.stat().st_size for f in target.rglob('*') if f.is_file()),
            construction_source_sa_bwt_bytes=700000122 if a.phase == 'produce' else 0,
            producer_and_verifier_source_bytes=sum((HERE/n).stat().st_size for n in ('producer.py', 'verifier.py')),
            query_results=replay['queries'], hostile_count=replay['hostile_rejections'],
            independent_index_source_semantics=False,
            limitations=['O(n) two-pass source verification per query; not succinct proof',
                'structural index validation plus historical digest pins, not independent LF/rank source-semantic proof',
                'unsigned external root; no freshness/rollback or Composition/O3/O4',
                'snapshot, ephemeral source and construction staging disk are nonzero',
                'finite CI artifact retention; not a full installed footprint ratio'])
        first_query = json.loads((target/'queries.json').read_bytes())[0]
        report['retained_runtime_code_identities'] = json.loads((target/'receipts'/(first_query['name']+'.json')).read_bytes())['runtime']['files']
        report['retained_runtime_source_bytes'] = sum(x['bytes'] for x in report['retained_runtime_code_identities'].values())
        report['additional_research_and_build_source_bytes'] = sum(f.stat().st_size for f in HERE.glob('*.py') if f.name not in ('producer.py', 'verifier.py')) + sum((LOC/n).stat().st_size for n in ('native_run_stream_inverse_rlb2_ef.cpp', 'native_run_interval_inverse_rlb2_ef.cpp', 'native_parallel_inverse_rlb2_ef.cpp', 'native_inverse_rlb2.cpp', 'core_l8_enwik8_pinned_materialize_gate.py', 'native_inverse_enwik8_gate.py', 'real_enwik8_ef_physical_gate.py', 'independent_materialization_replay_v0.py'))
        (target/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        print(json.dumps(report, sort_keys=True), flush=True)
        print('CORE_L9S_'+replay['status']+'_PASS', flush=True)
    finally:
        oracle.unlink(missing_ok=True)
        (target/'commands.json').write_text(json.dumps(commands, sort_keys=True, indent=2)+'\n')


if __name__ == '__main__':
    main()
