#!/usr/bin/env python3
"""Independent scanner tests, including real 1MiB boundaries and overlaps."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('standalone_scan_under_test', HERE/'verifier.py')
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    path = a.output/'ephemeral-test-source'
    chunk = 1024**2
    # Deliberately avoid a repeated >256 pattern at the boundary; this checks
    # exact canonical offsets, not merely a capped subset or total frequency.
    data = b'x'*(chunk-2)+b'\x00\xffABCDE'+b'x'*(chunk-3)+b'\x00\xffABCDE'+b'aaaaaaa'
    path.write_bytes(data)
    record = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    cases = []
    for pattern in (b'\x00\xffABCDE', b'\xffAB', b'BCDE', b'a', b'aa', b'aaa', b'absent', b'x', b'\x00', b'\xff'):
        for size in (chunk, 4*chunk):
            w = v.Work()
            observed = v.scan_authenticated(path, record, pattern, size, w)
            # Independent in-memory oracle is TEST ONLY, never a production verifier input.
            if pattern == b'x':
                expected_count, offsets = 2*chunk-5, list(range(256))
            else:
                offsets = [i for i in range(len(data)) if data.startswith(pattern, i)]
                expected_count = len(offsets)
            assert observed['count'] == expected_count and observed['offsets'] == offsets[:256]
            assert observed['scan_bytes'] == len(data) and observed['eof'] is True
            assert w.source_read_bytes == 2*len(data)
            cases.append(dict(pattern_hex=pattern.hex(), chunk_bytes=size, count=expected_count))
    rejected = []
    for name, changed, identity in [
        ('truncated-final-source', data[:-1], record),
        ('modified-final-source', data[:-1]+b'z', record),
        ('wrong-source-sha', data, dict(record, sha256='0'*64)),
        ('partial-scan-repinned-prefix-not-original-source', data[:chunk], record)]:
        path.write_bytes(changed)
        try:
            v.scan_authenticated(path, identity, b'\x00\xffABCDE', chunk, v.Work())
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('source mutation accepted: '+name)
    # A source changed after its authentication pass must also be rejected.
    # Use an actual file mutation via the production signature check, no mock
    # source or alternate scanner implementation.
    path.write_bytes(data)
    with v.opened(path, 100000000) as fd:
        before = v.signature(fd)
        with path.open('r+b') as out:
            out.seek(-1, 2)
            out.write(b'z')
        assert v.signature(fd) != before
        rejected.append('real-file-in-place-change-detected-by-signature')
    path.unlink()
    report = dict(status='STREAM_BOUNDARY_OVERLAP_PASS', positive_cases=len(cases), cases=cases,
                  rejected_source_mutations=rejected, boundary_source_bytes=len(data),
                  no_quadratic_sa=True, scan_buffer_sizes=[chunk, 4*chunk])
    (a.output/'stream-result.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
