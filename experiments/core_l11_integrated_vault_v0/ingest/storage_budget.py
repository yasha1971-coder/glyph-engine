"""Acceptance gate: count the complete retained candidate, never payload alone.

This does not publish, delete, modify or authenticate a candidate. Byte-perfect
replay and caller-authenticated source size are separate required gates.
"""
import argparse
import json
import stat
from pathlib import Path


def assess(candidate, source_bytes):
    if type(source_bytes) is not int or source_bytes < 0:
        raise ValueError('invalid externally established source byte count')
    candidate = Path(candidate)
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ValueError('candidate must be a real directory')
    total = 0
    files = 0
    for p in candidate.rglob('*'):
        s = p.lstat()
        if stat.S_ISDIR(s.st_mode):
            continue
        if not stat.S_ISREG(s.st_mode):
            raise ValueError('nonregular candidate entry: ' + str(p))
        total += s.st_size
        files += 1
    if not files:
        raise ValueError('empty candidate is not a preservation result')
    return dict(format='GLYPH_RETAINED_STORAGE_BUDGET_V1',
                source_bytes=source_bytes, retained_bytes=total, files=files,
                storage_budget_pass=total <= source_bytes,
                status='BUDGET_PASS' if total <= source_bytes else 'BUDGET_REJECTED',
                byte_perfect_replay_established=False,
                scope='all regular file lengths under candidate; include all required sidecars/indexes/history here; installed code and allocated blocks separate')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--source-bytes', type=int, required=True)
    a = p.parse_args()
    r = assess(a.candidate, a.source_bytes)
    print(json.dumps(r, sort_keys=True))
    raise SystemExit(0 if r['storage_budget_pass'] else 3)
