#!/usr/bin/env python3
"""Construction-only subprocess; exits before query processes start."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'


def main():
    p = argparse.ArgumentParser()
    for name in ('reference', 'decoder', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--repeat-budget-fixture', action='store_true')
    a = p.parse_args()
    ref, target = a.reference.resolve(), a.output.resolve()
    assert not target.exists(), 'new output required'
    assert subprocess.check_output(['git', '-C', str(ref), 'rev-parse', 'HEAD'], text=True).strip() == REF
    subprocess.run(['git', '-C', str(ref), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    target.mkdir(parents=True)
    construction = target/'construction'
    construction.mkdir()
    state, oracle = target/'state', target/'oracle'
    state.mkdir()
    oracle.mkdir()
    loc = Path(__file__).resolve().parents[1]/'loc_text_ef_v0'
    sys.path.insert(0, str(loc))
    from real_enwik8_ef_physical_gate import build, verify
    spec = importlib.util.spec_from_file_location('frozen_fixture', ref/'tools/check_rlbwt_query_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    f = module.build_fixture(construction, ref/'tools/rlbwt_container_v2.py', ref/'tools/rlbwt_rank_v2.py')
    assert f['source'] == b'\x00banana_bandana\xff\x00banana!' and len(f['source']) == 24
    if a.repeat_budget_fixture:
        # Supplemental budget boundary, constructed with the same frozen
        # encoder/rank builder. It does not replace the required 24-byte case.
        source = b'a'*257
        mapped = tuple(x+1 for x in source)+(0,)
        suffixes = sorted(range(len(mapped)), key=lambda i: mapped[i:])
        symbols = [256 if i == 0 else source[i-1] for i in suffixes]
        payload = struct.pack('<'+'H'*len(symbols), *symbols)
        bwt = construction/'repeat.bwt'
        bwt.write_bytes(struct.pack('<8sIQQIIIQQ', b'GLYBWT1\0', 1, len(source), len(mapped),
                                   257, 256, 16, len(payload), module.fnv1a64(payload))+payload)
        for name in ('rlb2', 'rlr2'):
            f[name].unlink()
        subprocess.run([sys.executable, str(ref/'tools/rlbwt_container_v2.py'), 'encode', str(bwt),
                        str(f['rlb2'])], check=True, capture_output=True)
        subprocess.run([sys.executable, str(ref/'tools/rlbwt_rank_v2.py'), 'build', str(f['rlb2']),
                        str(f['rlr2']), '--rank-step', '4'], check=True, capture_output=True)
        f.update(source=source, suffix_array=suffixes)
    sa = construction/'temporary.sa'
    sa.write_bytes(b'\0'*60+struct.pack('<'+'I'*len(f['suffix_array']), *f['suffix_array']))
    ef = construction/'physical.ef'
    geometry = build(sa, len(f['suffix_array']), f['sample_step'], ef)
    assert verify(sa, ef, geometry)['sampled_sa_records_verified'] > 0
    for name, path in [('rlb2', f['rlb2']), ('rlr2', f['rlr2']), ('ef', ef), ('decoder', a.decoder)]:
        shutil.copy2(path, state/name)
    def identity(path):
        raw = path.read_bytes()
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    root = dict(schema='GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0', frozen_commit=REF,
                source=dict(bytes=len(f['source']), sha256=hashlib.sha256(f['source']).hexdigest()),
                artifacts={n: identity(state/n) for n in ('rlb2', 'rlr2', 'ef')},
                decoder=dict(sha256=identity(state/'decoder')['sha256']))
    raw = (json.dumps(root, sort_keys=True, separators=(',', ':'))+'\n').encode()
    (state/'root.json').write_bytes(raw)
    (target/'pin.txt').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    # Delete every construction artifact, not just the temporary SA. No query
    # process inherits fixture source/SA objects from this construction process.
    removed = sorted(str(x.relative_to(construction)) for x in construction.iterdir())
    shutil.rmtree(construction)
    assert set(p.name for p in state.iterdir()) == {'root.json', 'rlb2', 'rlr2', 'ef', 'decoder'}
    args = [sys.executable, str(loc/'trusted_materialize_v0.py'), '--manifest', str(state/'root.json'),
            '--root-sha256', (target/'pin.txt').read_text().strip(), '--rlb2', str(state/'rlb2'),
            '--rlr2', str(state/'rlr2'), '--ef', str(state/'ef'), '--decoder', str(state/'decoder'),
            '--output', str(oracle/'materialized'), '--threads', '2', '--mode', 'rank']
    result = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert (oracle/'materialized').read_bytes() == f['source']
    (oracle/'l8-receipt.json').write_text(result.stdout)
    (target/'construction-audit.json').write_text(json.dumps(dict(
        removed_before_query=removed, retained_state_files=sorted(p.name for p in state.iterdir()),
        source_input_to_query=False, materialized_oracle_bytes=len(f['source'])), sort_keys=True)+'\n')
    print('CORE_L9_CONSTRUCTION_REMOVED_L8_MATERIALIZATION_PASS')


if __name__ == '__main__':
    main()
