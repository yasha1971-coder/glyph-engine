#!/usr/bin/env python3
"""Small, bounded CI orchestration; no dependencies, downloads or large corpora."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LOC = HERE.parents[0]/'loc_text_ef_v0'


def run(script, *args):
    subprocess.run([sys.executable, '-I', str(script), *map(str, args)], check=True, timeout=180)


def materialize(target):
    state, oracle = target/'state', target/'oracle'
    oracle.mkdir()
    # The inherited L8 executable has explicit local imports; run as designed.
    subprocess.run([sys.executable, str(LOC/'trusted_materialize_v0.py'),
        '--manifest', str(state/'root.json'), '--root-sha256', (target/'pin.txt').read_text().strip(),
        '--rlb2', str(state/'rlb2'), '--rlr2', str(state/'rlr2'), '--ef', str(state/'ef'),
        '--decoder', str(state/'decoder'), '--output', str(oracle/'materialized'),
        '--threads', '2', '--mode', 'rank'], check=True, timeout=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('exact', 'hostile'), required=True)
    for name in ('reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--capsule', type=Path)
    a = p.parse_args()
    target, ref = a.output.resolve(), a.reference.resolve()
    assert not target.exists(), 'new output required'
    if a.phase == 'exact':
        assert a.decoder is not None and a.capsule is None
        run(HERE/'fixture.py', '--reference', ref, '--decoder', a.decoder, '--output', target)
        budget = target/'budget'
        run(HERE/'fixture.py', '--reference', ref, '--decoder', a.decoder, '--output', budget,
            '--repeat-budget-fixture')
        run(HERE/'exact_gate.py', '--reference', ref, '--fixture', target, '--budget-fixture', budget)
        # Explicit bounded replay capsule: root/index/backend/receipts only.
        # The independent CI job must re-materialize its own authenticated
        # oracle; no raw source/SA/BWT is uploaded or retained for query.
        capsule = target/'capsule'
        capsule.mkdir()
        shutil.copytree(target/'state', capsule/'state')
        for name in ('pin.txt', 'found.json', 'empty.json', 'exact-result.json', 'construction-audit.json'):
            shutil.copy2(target/name, capsule/name)
        files = {str(path.relative_to(capsule)): dict(bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in sorted(capsule.rglob('*')) if path.is_file()}
        total = sum(x['bytes'] for x in files.values())
        assert total < 1024**2
        (capsule/'capsule-inventory.json').write_text(json.dumps(dict(
            payload_file_bytes=total, upload_cap_bytes=1024**2, files=files,
            raw_source_archive_bytes=0), sort_keys=True, indent=2)+'\n')
        assert sum(x.stat().st_size for x in capsule.rglob('*') if x.is_file()) < 1024**2
    else:
        assert a.capsule is not None and a.decoder is None
        capsule = a.capsule.resolve()
        expected = {'state', 'pin.txt', 'found.json', 'empty.json', 'exact-result.json',
                    'construction-audit.json', 'capsule-inventory.json'}
        assert set(x.name for x in capsule.iterdir()) == expected
        inventory = json.loads((capsule/'capsule-inventory.json').read_bytes())
        assert sum(x.stat().st_size for x in capsule.rglob('*') if x.is_file()) < 1024**2
        actual = {str(x.relative_to(capsule)) for x in capsule.rglob('*') if x.is_file()
                  and x.name != 'capsule-inventory.json'}
        assert actual == set(inventory['files'])
        for name, record in inventory['files'].items():
            raw = (capsule/name).read_bytes()
            assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
        shutil.copytree(capsule, target)
        # upload-artifact does not preserve executable permission bits.
        (target/'state/decoder').chmod(0o700)
        materialize(target)
        run(HERE/'hostile_gate.py', '--reference', ref, '--fixture', target)


if __name__ == '__main__':
    main()
