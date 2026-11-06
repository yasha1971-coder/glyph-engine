#!/usr/bin/env python3
"""Standalone CORE-L9 source-scan verifier. Standard library ONLY.

No producer/EF/FM/decoder imports or execution. Complete source is an explicit,
bounded L8 materialization input, authenticated BEFORE independent exact search.
The caller supplies the expected pattern, root and limit independently of receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import stat
import struct
import subprocess
import sys
import time

REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
RESEARCH_FILES = ('experiments/core_l9_exact_query_v0/producer.py',
                 'experiments/core_l9_exact_query_v0/verifier.py',
                 'experiments/loc_text_ef_v0/trusted_materialize_v0.py',
                 'experiments/loc_text_ef_v0/query_ef_research.py',
                 'experiments/loc_text_ef_v0/self_index_exact_range_extractor.py')
REFERENCE_FILES = ('tools/rlbwt_query_v2.py', 'tools/rlbwt_container_v2.py', 'tools/rlbwt_rank_v2.py')


def check(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False)+'\n').encode('ascii')


def unique(pairs):
    d = {}
    for k, v in pairs:
        check(k not in d, 'duplicate JSON field')
        d[k] = v
    return d


def decode(raw):
    x = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    check(type(x) is dict and canonical(x) == raw, 'noncanonical JSON')
    return x


def integer(x, high):
    return type(x) is int and 0 <= x <= high


def sha_string(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def keys(x, expected):
    check(type(x) is dict and set(x) == set(expected), 'unexpected/missing fields')


class Work:
    def __init__(self):
        self.hashed_bytes = self.sha_operations = self.source_read_bytes = 0

    def hash(self, raw):
        self.hashed_bytes += len(raw)
        self.sha_operations += 1
        return hashlib.sha256(raw).hexdigest()

    def read(self, path, cap):
        path = Path(path)
        check(stat.S_ISREG(path.lstat().st_mode) and path.stat().st_size <= cap,
              'nonregular/oversize input')
        # Read once into immutable bounded bytes; all validation/search uses this snapshot.
        with path.open('rb') as f:
            raw = f.read(cap+1)
        check(len(raw) <= cap, 'input grew beyond cap')
        return raw

    def object(self, path, record, cap):
        keys(record, ('bytes', 'sha256'))
        check(integer(record['bytes'], cap) and record['bytes'] > 0 and sha_string(record['sha256']),
              'bad object identity')
        raw = self.read(path, cap)
        check(len(raw) == record['bytes'] and self.hash(raw) == record['sha256'], 'object hash/size mismatch')
        return raw


def physical_oracle(source, components, w):
    """Independent, bounded small-source consistency check, NOT a query runtime.

    Derive an ephemeral SA/BWT oracle AFTER source authentication. No producer,
    FM, EF, container or decoder imports. This quadratic oracle is deliberately
    restricted to <=4096 bytes; it does not establish scalable index validation.
    """
    def fnv(raw):
        result = 14695981039346656037
        for byte in raw:
            result = ((result ^ byte)*1099511628211) & ((1 << 64)-1)
        return result

    n = len(source)+1
    mapped = tuple(b+1 for b in source)+(0,)
    sa = sorted(range(n), key=lambda i: mapped[i:])
    bwt = [256 if i == 0 else source[i-1] for i in sa]
    raw_bwt = struct.pack('<'+'H'*n, *bwt)
    b = components['rlb2']
    h = struct.unpack_from('<8sIIQQQIIIIIIQQQQ32s32s', b)
    check(h[:5] == (b'GLYRLB2\0', 2, 160, n-1, n) and h[6:8] == (257, 256)
          and h[9:12] == (1, 1, 0) and h[8] <= 255, 'RLB2 header/geometry')
    payload = b[160:]
    check(h[12] == len(payload) and h[13] == len(raw_bwt) and h[14] == fnv(raw_bwt)
          and h[15] == fnv(payload) and h[17].hex() == w.hash(payload), 'RLB2 checksums')
    bwt_header = struct.pack('<8sIQQIIIQQ', b'GLYBWT1\0', 1, n-1, n, 257, 256, 16,
                             len(raw_bwt), fnv(raw_bwt))
    check(h[16].hex() == w.hash(bwt_header+raw_bwt), 'construction BWT identity')
    runs, decoded, at = [], [], 0
    while at < len(payload):
        stream = at
        symbol = payload[at]
        at += 1
        if symbol == h[8]:
            check(at < len(payload) and payload[at] in (0, 1), 'run escape')
            symbol = h[8] if payload[at] == 0 else 256
            at += 1
        length, shift, digits = 0, 0, 0
        while True:
            check(at < len(payload) and digits < 10, 'run length overflow/truncation')
            byte = payload[at]
            at += 1
            digits += 1
            length |= (byte & 127) << shift
            if byte < 128:
                check(digits == 1 or byte != 0, 'noncanonical run length')
                break
            shift += 7
        check(0 < length <= n-len(decoded), 'invalid run length')
        check(not runs or runs[-1][2] != symbol, 'adjacent duplicate runs')
        runs.append((len(decoded), stream, symbol, length))
        decoded.extend([symbol]*length)
    check(decoded == bwt and len(runs) == h[5], 'RLB2 differs from verified source BWT')
    r = components['rlr2']
    rh = struct.unpack_from('<8sIIQQIIIIQIIQ32sQQ32sQ', r)
    step = rh[5]
    check(rh[:5] == (b'GLYRLR2\0', 2, 160, n, len(runs)) and 0 < step <= 2**32-1
          and rh[6:9] == (257, 256, 32) and rh[10:12] == (1052, 0), 'RLR2 geometry')
    positions = list(range(0, n, step))+[n]
    check(rh[9] == len(positions) and rh[12] == len(b) and rh[13].hex() == w.hash(b)
          and rh[14] == len(r)-160 == len(positions)*1052 and rh[15] == fnv(r[160:])
          and rh[16].hex() == w.hash(r[160:]) and rh[17] == fnv(r[:152]), 'RLR2 checksums')
    for index, pos in enumerate(positions):
        counts = [bwt[:pos].count(symbol) for symbol in range(257)]
        if pos == n:
            stream, offset = len(payload), 0
        else:
            run = next(run for run in runs if run[0] <= pos < run[0]+run[3])
            stream, offset = run[1], pos-run[0]
        expected = struct.pack('<QQQ257I', pos, stream, offset, *counts)
        check(r[160+index*1052:160+(index+1)*1052] == expected, 'RLR2 checkpoint/source mismatch')
    ef = components['ef']
    eh = struct.unpack_from('<8sQQQQBBHQQQQI', ef)
    step = eh[2]
    check(eh[0] == b'GL3EF001' and eh[1] == n and step > 0, 'EF geometry')
    samples = [(row, value//step) for row, value in enumerate(sa) if value % step == 0]
    m = len(samples)
    low, width = (n//m).bit_length()-1, (m-1).bit_length()
    highbits = ((n-1) >> low)+m+1
    lbs, hbs, qbs, cbs = (m*low+7)//8, (highbits+7)//8, (m*width+7)//8, 4*((m+255)//256)
    expected_header = struct.pack('<8sQQQQBBHQQQQI', b'GL3EF001', n, step, m,
                                  highbits, low, width, 256, lbs, hbs, qbs, cbs, 0)
    lows = highs = quotients = 0
    checkpoints = []
    for i, (row, quotient) in enumerate(samples):
        lows |= (row & ((1 << low)-1)) << (i*low)
        pos = (row >> low)+i
        highs |= 1 << pos
        quotients |= quotient << (i*width)
        if i % 256 == 0:
            checkpoints.append(pos)
    expected = (expected_header+lows.to_bytes(lbs, 'little')+highs.to_bytes(hbs, 'little')
                +quotients.to_bytes(qbs, 'little')+struct.pack('<'+'I'*len(checkpoints), *checkpoints))
    check(ef == expected, 'EF physical samples/geometry differ from source SA')


def verify(a):
    start, cpu = time.monotonic(), time.process_time()
    w = Work()
    check(sha_string(a.root_sha256), 'bad external pin')
    raw = w.read(a.manifest, 8192)
    check(w.hash(raw) == a.root_sha256, 'wrong external root')
    root = decode(raw)
    keys(root, ('schema', 'frozen_commit', 'source', 'artifacts', 'decoder'))
    check(root['schema'] == 'GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0'
          and root['frozen_commit'] == REF, 'wrong root version')
    keys(root['artifacts'], ('rlb2', 'rlr2', 'ef'))
    components = {name: w.object(getattr(a, name), root['artifacts'][name], 1024**2)
                  for name in ('rlb2', 'rlr2', 'ef')}
    keys(root['decoder'], ('sha256',))
    check(sha_string(root['decoder']['sha256']), 'invalid decoder identity')
    check(w.hash(w.read(a.decoder, 1024**2)) == root['decoder']['sha256'], 'decoder code identity')
    source = w.object(a.materialized, root['source'], 4096)
    w.source_read_bytes = len(source)
    physical_oracle(source, components, w)
    r = decode(w.read(a.receipt, 65536))
    keys(r, ('schema', 'trusted_root_sha256', 'source', 'artifacts', 'decoder', 'runtime',
             'scope', 'pattern_hex', 'pattern_bytes', 'count', 'policy', 'coverage_complete',
             'offsets', 'excerpts', 'status', 'reason'))
    check(r['schema'] == 'GLYPH_CORE_L9_EXACT_QUERY_RECEIPT_V0' and
          r['trusted_root_sha256'] == a.root_sha256 and r['source'] == root['source'] and
          r['artifacts'] == root['artifacts'] and r['decoder'] == root['decoder'], 'receipt/root identity')
    # Python numeric equality alone is insufficient: true and 1 must not alias identities.
    check(canonical(r['source']) == canonical(root['source']) and
          canonical(r['artifacts']) == canonical(root['artifacts']), 'typed identity mismatch')
    check(r['scope'] == 'single-complete-corpus; corpus-byte-offsets; no-doc-map; no-currentness', 'scope')
    pattern = bytes.fromhex(a.pattern_hex)
    check(0 < len(pattern) <= 4097 and pattern.hex() == a.pattern_hex, 'invalid expected pattern')
    check(r['pattern_hex'] == a.pattern_hex and type(r['pattern_bytes']) is int
          and r['pattern_bytes'] == len(pattern), 'changed query')
    check(integer(a.limit, 256), 'invalid expected policy')
    keys(r['policy'], ('mode', 'limit', 'order'))
    check(type(r['policy']['limit']) is int and r['policy']['limit'] == a.limit
          and r['policy']['order'] == 'ascending-corpus-byte-offset', 'changed query policy')
    # Full independent exact byte scan; overlap is intentional.
    expected = [i for i in range(len(source)) if source.startswith(pattern, i)]
    check(type(r['count']) is int and r['count'] == len(expected), 'count disagreement')
    check(type(r['offsets']) is list and all(integer(i, len(source)-1) for i in r['offsets']), 'offset types')
    check(r['coverage_complete'] is True and r['policy']['mode'] == 'full'
          and len(expected) <= a.limit and r['offsets'] == expected, 'incomplete/incorrect offsets')
    check(r['status'] == ('FOUND' if expected else 'PROVEN_EMPTY')
          and r['reason'] == 'verified-single-corpus', 'forged status')
    check(type(r['excerpts']) is list and len(r['excerpts']) == len(expected), 'missing excerpts')
    for off, e in zip(expected, r['excerpts']):
        keys(e, ('offset', 'bytes', 'sha256', 'bytes_included'))
        check(type(e['offset']) is int and type(e['bytes']) is int and e['offset'] == off
              and e['bytes'] == len(pattern) and e['bytes_included'] is False
              and e['sha256'] == w.hash(source[off:off+len(pattern)]), 'forged excerpt')
    reference = Path(a.reference)
    check(subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip() == REF,
          'wrong reference checkout')
    records = {}
    for prefix, folder, paths in [('research', Path(a.research), RESEARCH_FILES),
                                  ('reference', reference, REFERENCE_FILES)]:
        for path in paths:
            raw = w.read(folder/path, 1024**2)
            if prefix == 'reference':
                check(raw == subprocess.check_output(['git', '-C', str(folder), 'show', REF+':'+path]),
                      'modified frozen code')
            records[prefix+'/'+path] = dict(bytes=len(raw), sha256=w.hash(raw))
    runtime = dict(profile='CORE_L9_FROZEN_RANKV2_PHYSICAL_EF_V0', files=records,
                   revision_sha256=w.hash(canonical(records)))
    check(canonical(r['runtime']) == canonical(runtime), 'runtime/verifier revision mismatch')
    return dict(status='VERIFIED_SMALL', claim=r['status'], source_read_bytes=w.source_read_bytes,
                sha256_operations=w.sha_operations, hashed_bytes=w.hashed_bytes,
                scan_positions=len(source), receipt_bytes=Path(a.receipt).stat().st_size,
                wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    p = argparse.ArgumentParser()
    for key in ('manifest', 'root-sha256', 'rlb2', 'rlr2', 'ef', 'decoder', 'materialized',
                'receipt', 'pattern-hex', 'research', 'reference'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--limit', type=int, default=256)
    a = p.parse_args()
    try:
        print(json.dumps(verify(a), sort_keys=True))
        return 0
    except Exception as e:
        print(json.dumps(dict(status='UNTRUSTED', reason=str(e)[:200]), sort_keys=True), file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main())
