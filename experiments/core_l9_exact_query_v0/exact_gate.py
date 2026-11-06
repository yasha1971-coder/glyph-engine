#!/usr/bin/env python3
"""Exhaustive small-fixture index queries vs authenticated byte-scan oracle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_producer():
    spec = importlib.util.spec_from_file_location('l9_producer', HERE/'producer.py')
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    return p


def replay_command(reference, target, receipt, pattern, limit=256):
    state = target/'state'
    args = [sys.executable, '-I', str(HERE/'verifier.py'), '--manifest', str(state/'root.json'),
            '--root-sha256', (target/'pin.txt').read_text().strip(), '--reference', str(reference),
            '--research', str(REPO), '--materialized', str(target/'oracle/materialized'),
            '--receipt', str(receipt), '--pattern-hex', pattern.hex(), '--limit', str(limit)]
    for key in ('rlb2', 'rlr2', 'ef', 'decoder'):
        args += ['--'+key, str(state/key)]
    return args


def main():
    ap = argparse.ArgumentParser()
    for name in ('reference', 'fixture'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--budget-fixture', type=Path, required=True)
    a = ap.parse_args()
    target, ref = a.fixture.resolve(), a.reference.resolve()
    start, cpu = time.monotonic(), time.process_time()
    p = load_producer()
    state = target/'state'
    root = json.loads((state/'root.json').read_bytes())
    source = (target/'oracle/materialized').read_bytes()
    assert len(source) == root['source']['bytes'] == 24
    assert hashlib.sha256(source).hexdigest() == root['source']['sha256']
    patterns = [source[i:j] for i in range(24) for j in range(i+1, 25)]
    assert len(patterns) == 300
    extra = [b'\x00', b'\xff', b'\xff\x00', b'banana', b'ana', source, b'!', b'absent',
             b'BANANA', b' ', b'  ', source+b'x', b'\x00\xff', b'\xff\xff', b'a\x00', b'na!']
    reports, excerpt_total, sizes, crosses = [], 0, [], 0
    child_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    q = p.QueryState(state/'root.json', (target/'pin.txt').read_text().strip(), state/'rlb2',
                     state/'rlr2', state/'ef', state/'decoder', ref)
    try:
        for index, pattern in enumerate(patterns+extra):
            r = q.query(pattern)
            expected = [i for i in range(len(source)) if source.startswith(pattern, i)]
            assert r['count'] == len(expected) and r['offsets'] == expected
            assert r['coverage_complete'] is True
            assert r['status'] == ('FOUND' if expected else 'PROVEN_EMPTY')
            for hit in r['excerpts']:
                actual, _ = q.extractor.extract(hit['offset'], hit['bytes'])
                assert actual == pattern == source[hit['offset']:hit['offset']+hit['bytes']]
                assert hashlib.sha256(actual).hexdigest() == hit['sha256']
                excerpt_total += 1
                crosses += (hit['offset']//q.extractor.step !=
                            (hit['offset']+hit['bytes']-1)//q.extractor.step)
            raw = p.canonical(r)
            assert raw == p.canonical(q.query(pattern)), 'nondeterministic receipt'
            receipt = target/'current-receipt.json'
            receipt.write_bytes(raw)
            result = subprocess.run(replay_command(ref, target, receipt, pattern),
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, (index, result.stderr)
            report = json.loads(result.stdout)
            assert report['claim'] == r['status'] and report['source_read_bytes'] == 24
            reports.append(report)
            sizes.append(len(raw))
        limited = []
        for limit in (0, 1, 2):
            r = q.query(b'a', limit)
            assert r['count'] == source.count(b'a') and len(r['offsets']) == limit
            assert r['status'] == 'UNTRUSTED' and r['coverage_complete'] is False
            receipt.write_bytes(p.canonical(r))
            result = subprocess.run(replay_command(ref, target, receipt, b'a', limit),
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 3 and json.loads(result.stderr)['status'] == 'UNTRUSTED'
            limited.append(dict(limit=limit, count=r['count'], exit_code=result.returncode))
        try:
            q.query(b'')
            raise AssertionError('empty query accepted')
        except ValueError:
            pass
        # Empty corpus roots are rejected by the inherited positive source-size contract.
        empty = json.loads((state/'root.json').read_bytes())
        empty['source']['bytes'] = 0
        raw = p.canonical(empty)
        try:
            p.l8.parse_manifest(raw, p.sha(raw))
            raise AssertionError('empty source accepted')
        except ValueError:
            pass
        for name, pattern in [('found', b'banana'), ('empty', b'absent')]:
            (target/(name+'.json')).write_bytes(p.canonical(q.query(pattern)))
        sample_ram = q.extractor.sample_rows_ram_bytes
    finally:
        q.close()
    budget = a.budget_fixture.resolve()
    bs = budget/'state'
    bq = p.QueryState(bs/'root.json', (budget/'pin.txt').read_text().strip(), bs/'rlb2', bs/'rlr2',
                      bs/'ef', bs/'decoder', ref)
    try:
        over = bq.query(b'a')
        assert over['count'] == 257 and over['offsets'] == []
        assert over['status'] == 'UNTRUSTED' and over['coverage_complete'] is False
        boundary = bq.query(b'aa')
        assert boundary['count'] == 256 and boundary['offsets'] == list(range(256))
        path = budget/'boundary.json'
        path.write_bytes(p.canonical(boundary))
        verified = subprocess.run(replay_command(ref, budget, path, b'aa'),
                                  capture_output=True, text=True, timeout=30)
        assert verified.returncode == 0, verified.stderr
        budget_verification = json.loads(verified.stdout)
        path.write_bytes(p.canonical(over))
        rejected = subprocess.run(replay_command(ref, budget, path, b'a'),
                                  capture_output=True, text=True, timeout=30)
        assert rejected.returncode == 3 and json.loads(rejected.stderr)['status'] == 'UNTRUSTED'
        path.unlink()
    finally:
        bq.close()
    assert crosses > 0
    receipt.unlink()
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    code_paths = list(p.CODE)+list(p.REF_CODE)
    retained_code = sum((REPO/path if path in p.CODE else ref/path).stat().st_size for path in code_paths)
    support = [(HERE/name) for name in ('fixture.py', 'exact_gate.py', 'hostile_gate.py', 'run_gate.py')]
    support += [REPO/'experiments/loc_text_ef_v0'/name for name in (
        'real_enwik8_ef_physical_gate.py', 'native_run_stream_inverse_rlb2_ef.cpp',
        'native_run_interval_inverse_rlb2_ef.cpp', 'native_parallel_inverse_rlb2_ef.cpp',
        'native_inverse_rlb2.cpp')]
    support += [ref/'tools/check_rlbwt_query_v2.py']
    report = dict(status='VERIFIED_SMALL', reference_commit=p.FROZEN,
        research_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        root_sha256=(target/'pin.txt').read_text().strip(), source_sha256=root['source']['sha256'],
        substring_cases=300, additional_cases=len(extra), count_offset_receipt_replays=len(reports),
        excerpt_checks=excerpt_total, cross_sample_excerpt_checks=crosses, deterministic_receipts=len(reports),
        limited_rejections=limited, empty_pattern_rejected=True, empty_source_rejected=True,
        real_index_budget_boundary=dict(full_count=256, full_excerpts=256, limited_count=257,
                                        limited_offsets=0, verifier_limited_exit=3,
                                        full_verification_work=budget_verification),
        receipt_bytes_min=min(sizes), receipt_bytes_max=max(sizes), receipt_bytes_cap=65536,
        retained_index_bytes=sum((state/n).stat().st_size for n in ('rlb2', 'rlr2', 'ef')),
        retained_root_bytes=(state/'root.json').stat().st_size, retained_runtime_source_bytes=retained_code,
        reconstruction_and_gate_support_source_bytes=sum(path.stat().st_size for path in support),
        reconstruction_and_gate_support_files={str(path.relative_to(REPO) if path.is_relative_to(REPO)
            else 'reference/'+str(path.relative_to(ref))): dict(bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in support},
        retained_decoder_bytes=(state/'decoder').stat().st_size, retained_raw_source_bytes=0,
        separate_verifier_oracle_bytes=24, ephemeral_sample_map_bytes=sample_ram,
        measured_fixture_disk_bytes=sum(x.stat().st_size for x in target.rglob('*') if x.is_file()),
        producer_private_snapshot_bytes=sum((state/n).stat().st_size for n in ('rlb2', 'rlr2', 'ef', 'decoder')),
        producer_transient_validation_bwt_bytes=56+2*(len(source)+1),
        producer_transient_rebuilt_rank_bytes=(state/'rlr2').stat().st_size,
        verifier_source_read_bytes=sum(r['source_read_bytes'] for r in reports),
        verifier_sha256_operations=sum(r['sha256_operations'] for r in reports),
        verifier_hashed_bytes=sum(r['hashed_bytes'] for r in reports),
        verifier_wall_seconds=sum(r['wall_seconds'] for r in reports),
        verifier_cpu_seconds=sum(r['cpu_seconds'] for r in reports),
        verifier_peak_rss_kib=max(r['peak_rss_kib'] for r in reports),
        gate_wall_seconds=time.monotonic()-start, gate_cpu_seconds=time.process_time()-cpu,
        children_cpu_seconds=child_after.ru_utime+child_after.ru_stime-child_cpu.ru_utime-child_cpu.ru_stime,
        gate_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limits=['4096-byte source cap; quadratic independent SA oracle', '256-hit enumeration cap',
                '65536-byte canonical receipt cap', 'single corpus; external unsigned root',
                'Python/interpreter/OS/stdlib not vendored; no total installed footprint claim',
                'disk measure is file lengths; not filesystem allocated blocks or sampled peak'])
    (target/'exact-result.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))
    print('CORE_L9_A_B_EXACT_SMALL_PASS')


if __name__ == '__main__':
    main()
