#!/usr/bin/env python3
"""Independent process/hostile driver; imports no producer/search modules."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('reference', 'state', 'oracle', 'queries', 'receipts', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--test-fixture', action='store_true')
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    alone = a.output/'standalone_verifier.py'
    shutil.copy2(HERE/'verifier.py', alone)
    imports = []
    for node in ast.walk(ast.parse(alone.read_text())):
        if isinstance(node, ast.Import):
            imports += [n.name.split('.')[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.append(node.module.split('.')[0])
    assert set(imports) <= sys.stdlib_module_names
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ('eval', 'exec', '__import__', 'compile') for n in ast.walk(ast.parse(alone.read_text())))
    state = a.state.resolve()
    root = json.loads((state/'root.json').read_bytes())
    pin = sha((state/'root.json').read_bytes())
    receipt_path = a.output/'current-receipt.json'
    base = dict(manifest=state/'root.json', root_sha256=pin, rlb2=state/'rlb2', rlr2=state/'rlr2',
        ef=state/'ef', decoder=state/'decoder', reference=a.reference.resolve(), research=REPO,
        materialized=a.oracle.resolve(), receipt=receipt_path, pattern_hex='', limit=256)
    records, positives = [], []

    def invoke(r, changes=None, expect=3, name=None, count_verified=False):
        receipt_path.write_bytes(r if type(r) is bytes else canonical(r))
        values = dict(base)
        values.update(changes or {})
        args = [sys.executable, '-I', str(alone)]
        for key, value in values.items():
            args += ['--'+key.replace('_', '-'), str(value)]
        if a.test_fixture:
            args.append('--test-fixture')
        result = subprocess.run(args, capture_output=True, text=True, cwd=a.output, timeout=180)
        assert result.returncode == expect, (name, result.returncode, result.stdout, result.stderr)
        parsed = json.loads(result.stdout if expect == 0 else result.stderr)
        expected_status = ('VERIFIED_SMALL_SCAN' if a.test_fixture else 'VERIFIED_100MB_SCAN') if expect == 0 else 'UNTRUSTED'
        assert parsed['status'] == expected_status
        if count_verified:
            assert parsed['count_verified'] is True and parsed['full_source_authenticated'] is True
        if name:
            records.append(dict(name=name, exit_code=result.returncode, result=parsed))
        return parsed

    queries = json.loads(a.queries.read_bytes())
    receipts = {}
    for query in queries:
        receipt = json.loads((a.receipts/(query['name']+'.json')).read_bytes())
        assert receipt['pattern_hex'] == query['pattern_hex']
        expected = 3 if receipt['status'] == 'UNTRUSTED' else 0
        result = invoke(receipt, dict(pattern_hex=query['pattern_hex'], limit=query['limit']),
                        expect=expected, count_verified=True)
        assert result['independent_count'] == receipt['count'] and result['scan_bytes'] == root['source']['bytes']
        assert result['source_read_bytes'] == 2*root['source']['bytes'] and result['eof'] is True
        assert result['peak_rss_kib'] <= 64*1024, 'standalone verifier exceeded 64MiB RSS budget'
        positives.append(dict(query=query, exit_code=expected, verifier=result, count=receipt['count'],
                              offsets=receipt['offsets'], receipt_bytes=len(canonical(receipt))))
        receipts[query['name']] = receipt
    good = receipts['multi-1']
    assert good['count'] >= 2 and good['status'] == 'FOUND'
    empty = receipts['verified-empty']
    base['pattern_hex'] = good['pattern_hex']

    def mutate(name, edit):
        r = copy.deepcopy(good)
        edit(r)
        invoke(r, name=name)

    mutate('forged-PROVEN_EMPTY-against-known-positive', lambda r: r.update(count=0, offsets=[], excerpts=[], status='PROVEN_EMPTY'))
    mutate('count-off-by-one', lambda r: r.update(count=r['count']+1))
    mutate('missing-offset', lambda r: r['offsets'].pop())
    mutate('duplicated-offset', lambda r: r['offsets'].append(r['offsets'][0]))
    mutate('unsorted-offsets', lambda r: r['offsets'].reverse())
    mutate('fake-excerpt-sha', lambda r: r['excerpts'][0].update(sha256='0'*64))
    mutate('wrong-excerpt-length', lambda r: r['excerpts'][0].update(bytes=1))
    mutate('wrong-pattern-identity', lambda r: r.update(pattern_hex=empty['pattern_hex']))
    mutate('wrong-pattern-length', lambda r: r.update(pattern_bytes=r['pattern_bytes']+1))
    mutate('false-coverage', lambda r: r.update(coverage_complete=False))
    mutate('changed-query-policy', lambda r: r['policy'].update(limit=0))
    mutate('wrong-source-sha-receipt', lambda r: r['source'].update(sha256='0'*64))
    mutate('bool-count', lambda r: r.update(count=True))
    mutate('unknown-receipt-field-signed-authority', lambda r: r.update(signed_authority=True))
    mutate('invented-document-coordinate', lambda r: r.update(document_id=1))
    mutate('runtime-verifier-revision', lambda r: r['runtime'].update(revision_sha256='0'*64))
    invoke(good, dict(root_sha256='0'*64), name='wrong-external-root-pin')
    invoke(empty, name='forged-empty-by-query-substitution')
    for key in ('rlb2', 'rlr2', 'ef', 'decoder', 'materialized'):
        invoke(empty, {key: a.output/'missing', 'pattern_hex': empty['pattern_hex']}, name='empty-with-missing-'+key)
    raw = canonical(good)
    for name, bad in [('duplicate-JSON-key', b'{"count":1,'+raw[1:]),
                      ('truncated-receipt', raw[:-4]), ('invalid-UTF8', raw[:-2]+b'\xff}\n'),
                      ('noncanonical-receipt', b' '+raw), ('oversize-receipt', b'x'*65537),
                      ('nonfinite-receipt', raw.replace(b'"count":'+str(good['count']).encode(), b'"count":NaN', 1))]:
        invoke(bad, name=name)
    truncated = a.output/'partial-source'
    with a.oracle.open('rb') as f:
        truncated.write_bytes(f.read(16))
    invoke(empty, dict(materialized=truncated, pattern_hex=empty['pattern_hex']), name='forged-empty-from-partial-scan-truncated-source')
    fifo = a.output/'source.fifo'
    os.mkfifo(fifo)
    invoke(empty, dict(materialized=fifo, pattern_hex=empty['pattern_hex']), name='nonregular-source-never-empty')
    fifo.unlink()
    # Mutate the real ephemeral final source in place, then restore and verify
    # its original bytes. No reference/frozen artifact is modified.
    with a.oracle.open('r+b') as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    try:
        invoke(empty, dict(pattern_hex=empty['pattern_hex']), name='modified-final-materialized-source')
    finally:
        with a.oracle.open('r+b') as f:
            f.seek(-1, 2)
            f.write(last)

    def repinned(name, key=None, edit=None, root_edit=None, producer_check=False):
        modified = copy.deepcopy(root)
        values = dict(pattern_hex=empty['pattern_hex'])
        if key:
            raw = bytearray((state/key).read_bytes())
            edit(raw)
            path = a.output/name
            path.write_bytes(raw)
            modified['artifacts'][key] = dict(bytes=len(raw), sha256=sha(raw))
            values[key] = path
        if root_edit:
            root_edit(modified)
        path = a.output/(name+'-root.json')
        path.write_bytes(canonical(modified))
        newpin = sha(path.read_bytes())
        values.update(manifest=path, root_sha256=newpin)
        r = copy.deepcopy(empty)
        r.update(trusted_root_sha256=newpin, source=modified['source'], artifacts=modified['artifacts'])
        invoke(r, values, name=name)
        if name == 'new-root-EF-select-checkpoint-regression':
            assert records[-1]['result']['reason'] == 'EF select checkpoint'
        if name == 'new-root-EF-padding':
            assert records[-1]['result']['reason'] == 'EF noncanonical padding'
        if producer_check:
            args = [sys.executable, '-I', str(HERE/'producer.py')]
            merged = dict(base)
            merged.update(values)
            for k in ('manifest', 'root_sha256', 'rlb2', 'rlr2', 'ef', 'decoder', 'reference', 'pattern_hex'):
                args += ['--'+k.replace('_', '-'), str(merged[k])]
            if a.test_fixture:
                args.append('--test-fixture')
            out = subprocess.run(args, capture_output=True, text=True, timeout=180)
            assert out.returncode == 3 and json.loads(out.stderr)['status'] == 'UNTRUSTED', (name, out.stdout, out.stderr)
            records[-1]['producer_exit_code'] = 3
            records[-1]['producer_reason'] = json.loads(out.stderr)['reason']
    repinned('new-root-wrong-source-sha', root_edit=lambda r: r['source'].update(sha256='0'*64))
    repinned('new-root-wrong-RLR2-size', root_edit=lambda r: r['artifacts']['rlr2'].update(bytes=1))
    repinned('new-root-wrong-RLR2-digest', root_edit=lambda r: r['artifacts']['rlr2'].update(sha256='0'*64))
    def checkpoint(b):
        old = struct.unpack_from('<I', b, len(b)-4)[0]
        struct.pack_into('<I', b, len(b)-4, old+1)
    repinned('new-root-EF-select-checkpoint-regression', 'ef', checkpoint, producer_check=True)
    def padding(b):
        h = struct.unpack_from('<8sQQQQBBHQQQQI', b)
        assert (h[3]*h[6]) % 8 != 0
        pos = 80+h[8]+h[9]+h[10]-1
        b[pos] |= 128
    repinned('new-root-EF-padding', 'ef', padding, producer_check=True)
    repinned('new-root-RLR2-zero-step', 'rlr2', lambda b: struct.pack_into('<I', b, 32, 0), producer_check=True)
    result = dict(status='VERIFIED_SMALL_SCAN' if a.test_fixture else 'VERIFIED_100MB_SCAN',
        queries=positives, hostile_rejections=len(records), hostiles=records,
        standalone_imports=sorted(set(imports)), standalone_no_project_imports=True,
        full_source_read_bytes=sum(c['verifier']['source_read_bytes'] for c in positives),
        positive_sha_operations=sum(c['verifier']['sha256_operations'] for c in positives),
        negative_source_read_bytes=sum(c['result']['source_read_bytes'] for c in records),
        negative_component_read_bytes=sum(c['result']['component_read_bytes'] for c in records),
        negative_sha_operations=sum(c['result']['sha256_operations'] for c in records),
        verifier_peak_rss_kib=max(c['verifier']['peak_rss_kib'] for c in positives),
        wall_seconds=time.monotonic()-started, independent_index_source_semantics=False)
    assert len(records) >= 15
    receipt_path.unlink()
    (a.output/'result.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)
    print('CORE_L9S_INDEPENDENT_HOSTILES_PASS', flush=True)


if __name__ == '__main__':
    main()
