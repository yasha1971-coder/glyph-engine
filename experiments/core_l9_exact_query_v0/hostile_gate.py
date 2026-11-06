#!/usr/bin/env python3
"""Named hostile replays against a copied, standalone stdlib-only verifier."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('reference', 'fixture'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    target, ref = a.fixture.resolve(), a.reference.resolve()
    start = time.monotonic()
    work = target/'hostile'
    work.mkdir()
    verifier = work/'standalone_verifier.py'
    shutil.copy2(HERE/'verifier.py', verifier)
    tree = ast.parse(verifier.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(x.name.split('.')[0] for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.append(node.module.split('.')[0])
    assert set(imports) <= sys.stdlib_module_names, imports
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ('eval', 'exec', '__import__', 'compile') for n in ast.walk(tree))
    state = target/'state'
    root = json.loads((state/'root.json').read_bytes())
    pin = (target/'pin.txt').read_text().strip()
    found, empty = (json.loads((target/(n+'.json')).read_bytes()) for n in ('found', 'empty'))
    receipt = work/'receipt.json'
    records = []
    producer_negatives = []
    common = dict(manifest=state/'root.json', root_sha256=pin, reference=ref, research=REPO,
                  materialized=target/'oracle/materialized', receipt=receipt, pattern_hex=b'banana'.hex(),
                  rlb2=state/'rlb2', rlr2=state/'rlr2', ef=state/'ef', decoder=state/'decoder', limit=256)

    def run(receipt_value=found, overrides=None, expected=3, label=''):
        values = dict(common)
        values.update(overrides or {})
        receipt.write_bytes(receipt_value if isinstance(receipt_value, bytes) else canonical(receipt_value))
        command = [sys.executable, '-I', str(verifier)]
        for key, value in values.items():
            command += ['--'+key.replace('_', '-'), str(value)]
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=30)
        assert result.returncode == expected, (label, result.returncode, result.stdout, result.stderr)
        parsed = json.loads(result.stdout if expected == 0 else result.stderr)
        assert parsed['status'] == ('VERIFIED_SMALL' if expected == 0 else 'UNTRUSTED'), (label, parsed)
        if label:
            records.append(dict(name=label, verifier_exit=result.returncode, reason=parsed.get('reason')))
        return parsed

    def producer(values, pattern=b'absent', expected=3):
        args = [sys.executable, '-I', str(HERE/'producer.py')]
        for key in ('manifest', 'root_sha256', 'rlb2', 'rlr2', 'ef', 'decoder', 'reference'):
            args += ['--'+key.replace('_', '-'), str(values.get(key, common[key]))]
        args += ['--pattern-hex', pattern.hex()]
        result = subprocess.run(args, cwd=work, capture_output=True, text=True, timeout=30)
        assert result.returncode == expected, ('producer', values, result.stdout, result.stderr)
        obj = json.loads(result.stdout if expected == 0 else result.stderr)
        assert obj['status'] == ('PROVEN_EMPTY' if expected == 0 else 'UNTRUSTED')
        if expected:
            producer_negatives.append(dict(overrides=sorted(values), pattern_hex=pattern.hex(),
                                           exit_code=result.returncode, reason=obj.get('reason')))
        return result.returncode

    # Positive controls MUST pass from a directory containing only this copied
    # verifier, receipts and mutations. The verifier imports no project code.
    run(expected=0)
    run(empty, dict(pattern_hex=b'absent'.hex()), expected=0)
    oracle = common['materialized']
    moved = oracle.with_name('withheld')
    oracle.rename(moved)
    try:
        producer({}, expected=0)  # Producer succeeds without any raw source path.
    finally:
        moved.rename(oracle)

    def mutate(name, edit, base=found, overrides=None):
        r = copy.deepcopy(base)
        edit(r)
        run(r, overrides, label=name)

    mutate('wrong-count', lambda r: r.update(count=r['count']+1))
    mutate('missing-hit', lambda r: r['offsets'].pop())
    mutate('added-fake-hit', lambda r: r['offsets'].append(23))
    mutate('duplicate-offset', lambda r: r['offsets'].append(r['offsets'][0]))
    mutate('unsorted-offsets', lambda r: r['offsets'].reverse())
    mutate('changed-pattern-external-query-binding', lambda r: r.update(pattern_hex=b'bandana'.hex()))
    mutate('wrong-pattern-length', lambda r: r.update(pattern_bytes=7))
    mutate('excerpt-sha', lambda r: r['excerpts'][0].update(sha256='0'*64))
    mutate('excerpt-offset', lambda r: r['excerpts'][0].update(offset=0))
    mutate('excerpt-length', lambda r: r['excerpts'][0].update(bytes=5))
    mutate('excerpt-included-flag', lambda r: r['excerpts'][0].update(bytes_included=True))
    mutate('wrong-root-receipt', lambda r: r.update(trusted_root_sha256='0'*64))
    mutate('wrong-source-sha-receipt', lambda r: r['source'].update(sha256='0'*64))
    mutate('changed-limit', lambda r: r['policy'].update(limit=1))
    mutate('changed-policy-mode', lambda r: r['policy'].update(mode='limited'))
    mutate('false-completeness', lambda r: r.update(coverage_complete=False))
    mutate('stale-or-multidoc-scope', lambda r: r.update(scope='all-documents-current'))
    mutate('runtime-revision', lambda r: r['runtime'].update(revision_sha256='0'*64))
    mutate('verifier-file-identity', lambda r: r['runtime']['files']['research/experiments/core_l9_exact_query_v0/verifier.py'].update(sha256='0'*64))
    mutate('unknown-field-doc-id', lambda r: r.update(document_id=1))
    mutate('bool-count', lambda r: r.update(count=True))
    mutate('bool-offset', lambda r: r['offsets'].__setitem__(0, True))
    mutate('bool-excerpt-length', lambda r: r['excerpts'][0].update(bytes=True))
    mutate('bool-source-size', lambda r: r['source'].update(bytes=True))
    mutate('unknown-nested-policy-field', lambda r: r['policy'].update(trusted=True))
    mutate('unknown-schema', lambda r: r.update(schema='GLYPH_CORE_L9_UNREVIEWED'))
    # Three independent empty-result forgery mechanisms, never a missing
    # component interpreted as a valid count=0.
    mutate('proven-empty-forgery-1-erase-positive-hits', lambda r: r.update(
        count=0, offsets=[], excerpts=[], status='PROVEN_EMPTY'))
    run(empty, dict(pattern_hex=b'banana'.hex()), label='proven-empty-forgery-2-query-substitution')
    run(empty, dict(pattern_hex=b'absent'.hex(), rlr2=work/'missing-rank'),
        label='proven-empty-forgery-3-missing-rank')
    run(empty, dict(pattern_hex=b'absent'.hex(), ef=work/'missing-ef'), label='empty-with-missing-ef')
    run(overrides=dict(root_sha256='0'*64), label='wrong-external-root')
    producer(dict(root_sha256='0'*64))
    run(overrides=dict(pattern_hex=''), label='empty-expected-query')
    producer({}, pattern=b'')
    raw = canonical(found)
    for name, changed in [('duplicate-key', b'{"count":2,'+raw[1:]),
                          ('nonfinite', raw.replace(b'"count":2', b'"count":NaN')),
                          ('invalid-utf8', raw[:-2]+b'\xff}\n'),
                          ('truncated-json', raw[:-4]), ('noncanonical-whitespace', b' '+raw),
                          ('oversize-receipt', b' '*65537)]:
        assert changed != raw
        run(changed, label=name)
    for key in ('rlb2', 'rlr2', 'ef', 'decoder', 'materialized', 'manifest'):
        original = Path(common[key]).read_bytes()
        bad = work/('bitflip-'+key)
        changed = bytearray(original)
        changed[-1] ^= 1
        bad.write_bytes(changed)
        run(overrides={key: bad}, label=key+'-bitflip')
        if key in ('rlb2', 'rlr2', 'ef', 'decoder', 'manifest'):
            producer({key: bad})
    for key in ('rlb2', 'rlr2', 'ef', 'decoder'):
        run(overrides={key: work/'absent'}, label=key+'-missing')
        producer({key: work/'absent'})
    link = work/'symlink'
    link.symlink_to(state/'ef')
    run(overrides=dict(ef=link), label='symlink-ef')
    producer(dict(ef=link))

    def repin(name, key, edit):
        changed = bytearray((state/key).read_bytes())
        edit(changed)
        component = work/name
        component.write_bytes(changed)
        newroot = copy.deepcopy(root)
        newroot['artifacts'][key] = dict(bytes=len(changed), sha256=sha(changed))
        rootpath = work/(name+'-root.json')
        rootpath.write_bytes(canonical(newroot))
        newpin = sha(rootpath.read_bytes())
        r = copy.deepcopy(empty)
        r.update(trusted_root_sha256=newpin, artifacts=newroot['artifacts'])
        values = {key: component, 'manifest': rootpath, 'root_sha256': newpin,
                  'pattern_hex': b'absent'.hex()}
        run(r, values, label=name)
        records[-1]['producer_exit'] = producer(values)

    repin('repinned-ef-zero-step', 'ef', lambda b: struct.pack_into('<Q', b, 16, 0))
    repin('repinned-ef-invalid-select-checkpoint', 'ef', lambda b: struct.pack_into('<I', b, len(b)-4, 1))
    repin('repinned-rank-zero-step', 'rlr2', lambda b: struct.pack_into('<I', b, 32, 0))
    repin('repinned-rank-bad-geometry', 'rlr2', lambda b: struct.pack_into('<Q', b, 48, 1))
    def wrong_counter(b):
        # Preserve all checksums to exercise deterministic rank rebuilding and
        # the independent source-derived checkpoint oracle, not hash rejection.
        struct.pack_into('<I', b, 160+1052+24, 1)
        payload = bytes(b[160:])
        def fnv(raw):
            value = 14695981039346656037
            for byte in raw:
                value = ((value ^ byte)*1099511628211) & ((1 << 64)-1)
            return value
        struct.pack_into('<Q', b, 112, fnv(payload))
        b[120:152] = hashlib.sha256(payload).digest()
        struct.pack_into('<Q', b, 152, fnv(b[:152]))
    repin('repinned-rank-counter-valid-checksums', 'rlr2', wrong_counter)
    # Original fixture: 80-byte header, zero low bytes, 5 high bytes,
    # then 52 quotient bits in 7 bytes. The final 4 quotient bits are padding.
    repin('repinned-ef-noncanonical-padding', 'ef', lambda b: b.__setitem__(91, b[91] | 128))
    newroot = copy.deepcopy(root)
    newroot['source']['sha256'] = '0'*64
    rootpath = work/'repinned-wrong-source.json'
    rootpath.write_bytes(canonical(newroot))
    r = copy.deepcopy(empty)
    r.update(trusted_root_sha256=sha(rootpath.read_bytes()), source=newroot['source'])
    values = dict(manifest=rootpath, root_sha256=r['trusted_root_sha256'], pattern_hex=b'absent'.hex())
    run(r, values, label='repinned-wrong-source-identity')
    records[-1]['producer_exit'] = producer(values)
    for label, rawroot in [('repinned-duplicate-root-key', b'{"schema":"bad",'+canonical(root)[1:]),
                           ('repinned-unknown-root-field', canonical(dict(root, authority=True))),
                           ('repinned-zero-source-length', canonical(dict(root, source=dict(bytes=0, sha256=root['source']['sha256']))))]:
        path = work/(label+'.json')
        path.write_bytes(rawroot)
        values = dict(manifest=path, root_sha256=sha(rawroot))
        run(overrides=values, label=label)
        records[-1]['producer_exit'] = producer(values)
    report = dict(status='VERIFIED_SMALL', hostile_rejections=len(records), cases=records,
                  producer_negative_rejections=len(producer_negatives), producer_cases=producer_negatives,
                  independent_positive_controls=2, producer_without_raw_source=True,
                  standalone_verifier_imports=sorted(set(imports)),
                  wall_seconds=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (target/'hostile-result.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))
    print('CORE_L9_INDEPENDENT_HOSTILE_REPLAY_PASS')


if __name__ == '__main__':
    main()
