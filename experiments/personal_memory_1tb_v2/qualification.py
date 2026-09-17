#!/usr/bin/env python3
"""Bounded Linux qualification measurements. Only reads an existing Golden demo."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent

def digest(b):
    return hashlib.sha256(b).hexdigest()

def summary(values):
    a = sorted(values)
    if not a:
        raise ValueError('empty measurements')
    return dict(n=len(a), minimum=a[0], median=statistics.median(a),
                p95=a[math.ceil(.95*len(a))-1], maximum=a[-1])

def receipt(root):
    p = root / 'GLYPH_GOLDEN_DEMO_RC1.json'
    raw = p.read_bytes()
    if digest(raw) != Path(str(p)+'.sha256').read_text().strip():
        raise ValueError('demo receipt checksum mismatch')
    r = json.loads(raw)
    if r['status'] != 'LOCAL_GATES_PASSED':
        raise ValueError('demo was not completed')
    return r

def worker(root, kind):
    import incremental_memory as inc
    r = receipt(root)
    if kind == 'deflate6':
        samples = []
        start = time.perf_counter()
        for f in r['source_files']:
            b = (root/'input'/f['archive_path']).read_bytes()
            if len(b) != f['bytes'] or digest(b) != f['sha256']:
                raise ValueError('baseline input mismatch')
            t = time.perf_counter(); encoded = zlib.compress(b,6)
            encode = time.perf_counter()-t
            t = time.perf_counter(); decoded = zlib.decompress(encoded)
            decode = time.perf_counter()-t
            if decoded != b: raise ValueError('baseline roundtrip mismatch')
            samples.append(dict(bytes=len(b), stored_bytes=len(encoded),
                                encode_seconds=encode, decode_seconds=decode))
            del b, encoded, decoded
        return dict(kind=kind, seconds=time.perf_counter()-start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    samples=samples)
    start = time.perf_counter()
    m = inc.Memory(root/'memory', root/'archive', r['base_receipt_sha256'])
    first = r['versions'][0]['snapshot']
    if kind == 'base':
        jobs = [(first, f['archive_path'], f['sha256'], f['bytes']) for f in r['source_files']]
    else:
        jobs = [(v['snapshot'], 'xml.xml', v['sha256'], v['bytes']) for v in r['versions']]
    samples = []
    for pin, path, expected, size in jobs:
        t = time.perf_counter()
        b = m.read_item(m.snapshot(pin)['files'][path])
        decode = time.perf_counter()-t
        if len(b) != size or digest(b) != expected:
            raise ValueError('restoration mismatch')
        samples.append(dict(bytes=size, read_seconds=decode,
                            verified_seconds=time.perf_counter()-t))
        del b
    return dict(kind=kind, seconds=time.perf_counter()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                samples=samples)

def launch(root, kind):
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', kind,
                        '--demo', str(root)], capture_output=True, text=True, timeout=600)
    if p.returncode:
        raise RuntimeError('measurement worker failed: '+p.stderr[-3000:])
    return json.loads(p.stdout)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--worker', choices=['base','history','deflate6'], help=argparse.SUPPRESS)
    a = p.parse_args()
    root = a.demo.resolve()
    if a.worker:
        print(json.dumps(worker(root, a.worker))); return
    if not 3 <= a.repeats <= 10:
        p.error('repeats must be 3..10')
    if a.output is None:
        p.error('--output required')
    out = a.output.resolve()
    if out == root or root in out.parents or out in root.parents:
        p.error('output must be separate from demo')
    r = receipt(root)
    # Output must be new; partial results are retained on any failure.
    out.mkdir(parents=False, exist_ok=False)
    def emit(stage, **kw):
        print(json.dumps(dict(stage=stage, **kw)), flush=True)
    initial = digest((root/'GLYPH_GOLDEN_DEMO_RC1.json').read_bytes())
    emit('regression-tests')
    with (out/'tests.txt').open('w') as log:
        done = subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-v'],
                              stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if done.returncode:
        raise RuntimeError('regression tests failed: see tests.txt')
    results = {}
    for kind in ('base', 'history', 'deflate6'):
        rows = []
        for i in range(a.repeats):
            row = launch(root,kind); rows.append(row)
            (out/f'{kind}-{i}.json').write_text(json.dumps(row,indent=2))
            emit('measured', workload=kind, repeat=i+1, seconds=row['seconds'])
        results[kind] = dict(batch_seconds=summary([x['seconds'] for x in rows]),
                            peak_rss_bytes=summary([x['peak_rss_bytes'] for x in rows]), runs=rows)
    # Two independent readers. Does not mutate the running application's catalogue.
    emit('concurrent-readers', readers=2)
    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        paired = list(ex.map(lambda _: launch(root,'base'), range(2)))
    concurrent_seconds = time.perf_counter()-t
    if digest((root/'GLYPH_GOLDEN_DEMO_RC1.json').read_bytes()) != initial:
        raise RuntimeError('receipt changed during measurement')
    report = dict(format='GLYPH_QUALIFICATION_V1', status='MEASURED_GATES_PASSED',
        demo_receipt_sha256=initial, demo_commit=r['code_commit'],
        sources_sha256={x.name:digest(x.read_bytes()) for x in sorted(HERE.glob('*.py'))},
        platform=platform.platform(), python=sys.version, cpu_count=os.cpu_count(),
        regressions_exit_code=done.returncode, tests_sha256=digest((out/'tests.txt').read_bytes()),
        repetitions=a.repeats, results=results,
        concurrent_base=dict(readers=2, wall_seconds=concurrent_seconds, runs=paired),
        methodology=['fresh process for each batch, OS caches not flushed',
                     'deflate6 uses separate zlib frames per file; excludes catalogue metadata',
                     'deflate6 decode timing excludes verification; GLYPH read timing includes internal checks',
                     'read includes snapshot lookup, decode and internal integrity checks',
                     'verified time additionally includes external SHA-256',
                     'batch includes Memory initialization and all reads; excludes process startup',
                     'Linux ru_maxrss is per-worker peak RSS; not whole-system memory',
                     'p95 nearest rank across batch repetitions; small sample, not a tail-latency SLA',
                     'tests include synthetic corruption, concurrency and journal recovery cases'],
        not_proven=['power-loss durability on physical hardware','one-terabyte scalability',
                    'native browser rendering latency','performance on iPhone','real-editor version workloads',
                    'industrial certification','cold disk performance'])
    raw = json.dumps(report,ensure_ascii=False,indent=2).encode()
    (out/'QUALIFICATION.json').write_bytes(raw)
    (out/'QUALIFICATION.json.sha256').write_text(digest(raw)+'\n')
    emit('qualification-complete', report=str(out/'QUALIFICATION.json'))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('STOP: '+str(e), file=sys.stderr)
        sys.exit(2)
