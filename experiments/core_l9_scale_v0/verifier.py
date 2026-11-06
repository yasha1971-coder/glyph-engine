#!/usr/bin/env python3
"""Standalone bounded-memory exact byte scan, standard library ONLY.

No producer/FM/EF/L8 imports or execution; no SA/BWT construction. Structural
index checks do NOT establish independent index-source LF/rank semantics.
Exact query evidence comes from a complete, twice-hashed source scan. Caller
supplies expected root/query/policy; root is unsigned, not freshness authority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import struct
import subprocess
import sys
import time

REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
SOURCE = '2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8'
PINS = {'rlb2': (73365105, '4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3'),
        'rlr2': (12844028, 'e76a49db554860bf907aa307a503240d1e71ef0400282c34548a9726a3f22719'),
        'ef': (2844324, 'c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7')}
RESEARCH_FILES = ('experiments/core_l9_scale_v0/producer.py', 'experiments/core_l9_scale_v0/verifier.py',
    'experiments/core_l9_exact_query_v0/producer.py', 'experiments/loc_text_ef_v0/trusted_materialize_v0.py',
    'experiments/loc_text_ef_v0/query_ef_research.py', 'experiments/loc_text_ef_v0/self_index_exact_range_extractor.py')
REFERENCE_FILES = ('tools/rlbwt_query_v2.py', 'tools/rlbwt_container_v2.py')


def check(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode('ascii')


def unique(pairs):
    d = {}
    for k, v in pairs:
        check(k not in d, 'duplicate JSON key')
        d[k] = v
    return d


def decode(raw):
    x = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    check(type(x) is dict and canonical(x) == raw, 'noncanonical JSON')
    return x


def keys(x, names):
    check(type(x) is dict and set(x) == set(names), 'unknown/missing JSON fields')


def integer(x, maximum):
    return type(x) is int and 0 <= x <= maximum


def sha_string(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def signature(fd):
    s = os.fstat(fd.fileno())
    check(stat.S_ISREG(s.st_mode), 'not a regular file')
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def opened(path, cap):
    info = os.lstat(path)
    check(stat.S_ISREG(info.st_mode) and info.st_size <= cap, 'bounded regular file required')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    stream = os.fdopen(fd, 'rb')
    try:
        check(signature(stream)[2] <= cap, 'input exceeds admission bound')
        return stream
    except BaseException:
        stream.close()
        raise


class Work:
    def __init__(self):
        self.sha_operations = self.hashed_bytes = self.source_read_bytes = self.component_read_bytes = 0

    def digest(self, raw):
        self.sha_operations += 1
        self.hashed_bytes += len(raw)
        return hashlib.sha256(raw).hexdigest()

    def small(self, path, cap):
        with opened(path, cap) as f:
            before = signature(f)
            raw = f.read(cap+1)
            check(len(raw) <= cap and signature(f) == before, 'changing/oversize bounded file')
        return raw

    def file(self, path, record, cap, payload=False):
        keys(record, ('bytes', 'sha256'))
        check(integer(record['bytes'], cap) and record['bytes'] > 0 and sha_string(record['sha256']), 'bad object identity')
        h, p = hashlib.sha256(), hashlib.sha256()
        self.sha_operations += 1+int(payload)
        size = 0
        with opened(path, cap) as f:
            before = signature(f)
            check(before[2] == record['bytes'], 'object size disagreement')
            for data in iter(lambda: f.read(1024**2), b''):
                check(size+len(data) <= record['bytes'], 'object grew')
                h.update(data)
                self.hashed_bytes += len(data)
                if payload:
                    tail = data[max(0, 160-size):]
                    p.update(tail)
                    self.hashed_bytes += len(tail)
                size += len(data)
                self.component_read_bytes += len(data)
            check(signature(f) == before, 'object changed during hash')
        check(size == record['bytes'] and h.hexdigest() == record['sha256'], 'object hash disagreement')
        return p.hexdigest() if payload else h.hexdigest()


def index_structure(a, root, payload_hashes, w):
    """O(rank checkpoints + EF samples) structural validation, bounded storage.

    Explicit limit: does not independently replay rank prefixes against BWT or
    EF suffix values against source. A correct source scan proves this query's
    result, not arbitrary FM-index semantics or an authenticated index proof.
    """
    n = root['source']['bytes']+1
    with opened(a.rlb2, 128*1024**2) as f:
        raw = f.read(160)
    h = struct.unpack('<8sIIQQQIIIIIIQQQQ32s32s', raw)
    check(h[:5] == (b'GLYRLB2\0', 2, 160, n-1, n) and 0 < h[5] <= n
          and h[6:8] == (257, 256) and h[8] <= 255 and h[9:12] == (1, 1, 0), 'RLB2 header geometry')
    check(h[12] == root['artifacts']['rlb2']['bytes']-160 and h[13] == 2*n
          and h[17].hex() == payload_hashes['rlb2'], 'RLB2 payload geometry/digest')
    with opened(a.rlr2, 128*1024**2) as f:
        before = signature(f)
        r = struct.unpack('<8sIIQQIIIIQIIQ32sQQ32sQ', f.read(160))
        step = r[5]
        check(r[:5] == (b'GLYRLR2\0', 2, 160, n, h[5]) and step > 0 and r[6:9] == (257, 256, 32)
              and r[10:12] == (1052, 0), 'RLR2 header geometry')
        check(r[9] == (n+step-1)//step+1 and r[12] == root['artifacts']['rlb2']['bytes']
              and r[13].hex() == root['artifacts']['rlb2']['sha256']
              and r[14] == root['artifacts']['rlr2']['bytes']-160 == r[9]*1052
              and r[16].hex() == payload_hashes['rlr2'], 'RLR2 binding/geometry/digest')
        previous, previous_stream = [0]*257, 0
        for i in range(r[9]):
            data = f.read(1052)
            w.component_read_bytes += len(data)
            pos, stream, off, *counts = struct.unpack('<QQQ257I', data)
            check(pos == (n if i == r[9]-1 else i*step) and sum(counts) == pos
                  and all(x <= y for x, y in zip(previous, counts)), 'RLR2 checkpoint prefix geometry')
            check(previous_stream <= stream <= h[12] and off <= pos, 'RLR2 checkpoint stream geometry')
            if i == 0:
                check(stream == off == 0, 'RLR2 initial checkpoint')
            if i == r[9]-1:
                check(stream == h[12] and off == 0 and counts[256] == 1, 'RLR2 terminal checkpoint')
            previous, previous_stream = counts, stream
        check(not f.read(1) and signature(f) == before, 'RLR2 trailing/changing file')
    # EF is <=4MiB for this specific 100MB geometry. No mmap-dependent RSS or
    # O(n) source table: bounded bitfields plus m-bit quotient coverage bitmap.
    ef = w.small(a.ef, 4*1024**2)
    w.component_read_bytes += len(ef)+320
    e = struct.unpack_from('<8sQQQQBBHQQQQI', ef)
    step, m, highbits, low, width, stride = e[2:8]
    check(e[0] == b'GL3EF001' and e[1] == n and step > 0 and m == (n-1)//step+1
          and low == (n//m).bit_length()-1 and width == (m-1).bit_length()
          and highbits == ((n-1) >> low)+m+1 and stride == 256 and e[12] == 0, 'EF header geometry')
    sizes = ((m*low+7)//8, (highbits+7)//8, (m*width+7)//8, 4*((m+255)//256))
    check(e[8:12] == sizes and len(ef) == 80+sum(sizes), 'EF section geometry')
    check(m <= 1000000, 'EF bounded sample bitmap admission')
    at = 80
    fields = []
    for size in sizes:
        fields.append(memoryview(ef)[at:at+size])
        at += size
    lows, high, quotients, checkpoints = fields
    for data, bits in ((lows, m*low), (high, highbits), (quotients, m*width)):
        check(not bits % 8 or data[-1] >> (bits % 8) == 0, 'EF noncanonical padding')
    def bits(data, pos, width):
        return (int.from_bytes(data[pos//8:(pos+width+7)//8], 'little') >> (pos % 8)) & ((1 << width)-1)
    seen = bytearray((m+7)//8)
    sample, last_row = 0, -1
    for byte_index, packed in enumerate(high):
        while packed:
            bit = packed & -packed
            pos = byte_index*8+bit.bit_length()-1
            check(sample < m and pos < highbits, 'EF bitmap coverage')
            row = ((pos-sample) << low) | bits(lows, sample*low, low)
            q = bits(quotients, sample*width, width)
            check(last_row < row < n and q < m and q*step < n and not seen[q//8] & (1 << (q % 8)),
                  'EF row/quotient geometry')
            seen[q//8] |= 1 << (q % 8)
            if sample % 256 == 0:
                check(struct.unpack_from('<I', checkpoints, 4*(sample//256))[0] == pos, 'EF select checkpoint')
            last_row, sample = row, sample+1
            packed ^= bit
    check(sample == m, 'EF sample cardinality')
    return dict(rank_checkpoints=r[9], ef_samples=m, independent_index_source_semantics=False,
                scope='structural-only; exact query justified by authenticated complete source scan')


def scan_authenticated(path, record, pattern, chunk_bytes, w):
    """Two source passes on the SAME regular FD, bounded buffers and hit list.

    Pass 1 authenticates before scan; pass 2 hashes again while scanning so a
    changed/truncated source cannot turn a prefix scan into a trusted empty.
    Matches overlap and cross chunk boundaries. Stored offsets never exceed 256.
    """
    check(1024**2 <= chunk_bytes <= 4*1024**2 and 0 < len(pattern) <= 4097, 'scan admission')
    keys(record, ('bytes', 'sha256'))
    check(integer(record['bytes'], 100000000) and record['bytes'] > 0 and sha_string(record['sha256']), 'source identity')
    with opened(path, 100000000) as f:
        before = signature(f)
        check(before[2] == record['bytes'], 'source size/truncation')
        digest, total = hashlib.sha256(), 0
        w.sha_operations += 1
        for data in iter(lambda: f.read(chunk_bytes), b''):
            total += len(data)
            w.source_read_bytes += len(data)
            w.hashed_bytes += len(data)
            digest.update(data)
            check(total <= record['bytes'], 'source grew while authenticating')
        check(total == record['bytes'] and digest.hexdigest() == record['sha256'] and signature(f) == before,
              'source SHA/signature changed before search')
        f.seek(0)
        digest, total, count, carry = hashlib.sha256(), 0, 0, b''
        offsets, excerpts = [], []
        w.sha_operations += 1
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            check(total+len(data) <= record['bytes'], 'source grew during scan')
            digest.update(data)
            w.hashed_bytes += len(data)
            w.source_read_bytes += len(data)
            window = carry+data
            start = 0
            while True:
                at = window.find(pattern, start)
                if at < 0:
                    break
                count += 1
                if len(offsets) < 256:
                    offset = total-len(carry)+at
                    offsets.append(offset)
                    excerpts.append(dict(offset=offset, bytes=len(pattern),
                                         sha256=w.digest(window[at:at+len(pattern)]), bytes_included=False))
                start = at+1
            total += len(data)
            carry = window[-(len(pattern)-1):] if len(pattern) > 1 else b''
        check(total == record['bytes'] and digest.hexdigest() == record['sha256'] and signature(f) == before,
              'source changed/truncated during full EOF scan')
        return dict(count=count, offsets=offsets, excerpts=excerpts, scan_bytes=total,
                    eof=True, source_sha256=digest.hexdigest(), truncated=count > 256)


def verify(a, w):
    check(sha_string(a.root_sha256), 'invalid external root')
    raw = w.small(a.manifest, 8192)
    check(w.digest(raw) == a.root_sha256, 'wrong external root')
    root = decode(raw)
    keys(root, ('schema', 'frozen_commit', 'source', 'artifacts', 'decoder'))
    check(root['schema'] == 'GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0' and root['frozen_commit'] == REF, 'root version')
    keys(root['source'], ('bytes', 'sha256'))
    check(integer(root['source']['bytes'], 100000000) and root['source']['bytes'] > 0 and sha_string(root['source']['sha256']), 'source root identity')
    keys(root['artifacts'], ('rlb2', 'rlr2', 'ef'))
    payloads = {name: w.file(getattr(a, name), root['artifacts'][name], 128*1024**2, name != 'ef')
                for name in ('rlb2', 'rlr2', 'ef')}
    keys(root['decoder'], ('sha256',))
    check(sha_string(root['decoder']['sha256']) and w.digest(w.small(a.decoder, 1024**2)) == root['decoder']['sha256'], 'decoder identity')
    geometry = index_structure(a, root, payloads, w)
    if a.test_fixture:
        check(root['source']['bytes'] <= 4096, 'synthetic opt-in is small only')
    else:
        check(root['source'] == dict(bytes=100000000, sha256=SOURCE), 'historical enwik8 source differs')
        for name, (size, digest) in PINS.items():
            check(root['artifacts'][name] == dict(bytes=size, sha256=digest), 'historical physical index differs')
    r = decode(w.small(a.receipt, 65536))
    keys(r, ('schema', 'trusted_root_sha256', 'source', 'artifacts', 'decoder', 'runtime', 'scope',
             'pattern_hex', 'pattern_bytes', 'count', 'policy', 'coverage_complete', 'offsets', 'excerpts', 'status', 'reason'))
    check(r['schema'] == 'GLYPH_CORE_L9_EXACT_QUERY_RECEIPT_V0' and r['trusted_root_sha256'] == a.root_sha256,
          'receipt root/schema')
    for key in ('source', 'artifacts', 'decoder'):
        check(canonical(r[key]) == canonical(root[key]), 'receipt identity differs')
    check(r['scope'] == 'single-complete-corpus; corpus-byte-offsets; no-doc-map; no-currentness', 'scope')
    pattern = bytes.fromhex(a.pattern_hex)
    check(0 < len(pattern) <= 4097 and pattern.hex() == a.pattern_hex and r['pattern_hex'] == a.pattern_hex
          and type(r['pattern_bytes']) is int and r['pattern_bytes'] == len(pattern), 'expected query identity')
    check(integer(a.limit, 256), 'expected policy limit')
    keys(r['policy'], ('mode', 'limit', 'order'))
    check(type(r['policy']['limit']) is int and r['policy']['limit'] == a.limit
          and r['policy']['order'] == 'ascending-corpus-byte-offset', 'expected policy mismatch')
    reference = Path(a.reference)
    check(subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip() == REF, 'frozen reference HEAD')
    files = {}
    for prefix, folder, names in [('research', Path(a.research), RESEARCH_FILES), ('reference', reference, REFERENCE_FILES)]:
        for name in names:
            data = w.small(folder/name, 1024**2)
            if prefix == 'reference':
                check(data == subprocess.check_output(['git', '-C', str(folder), 'show', REF+':'+name]), 'dirty frozen source')
            files[prefix+'/'+name] = dict(bytes=len(data), sha256=w.digest(data))
    runtime = dict(profile='CORE_L9S_STREAM_SCAN_V0', files=files, revision_sha256=w.digest(canonical(files)))
    check(canonical(runtime) == canonical(r['runtime']), 'runtime/verifier revision mismatch')
    scan = scan_authenticated(a.materialized, root['source'], pattern, a.chunk_bytes, w)
    count = scan['count']
    check(type(r['count']) is int and r['count'] == count, 'independent count disagreement')
    check(type(r['offsets']) is list and all(integer(i, root['source']['bytes']-1) for i in r['offsets']), 'offset types')
    offsets = scan['offsets'][:a.limit] if count <= 256 else []
    excerpts = scan['excerpts'][:a.limit] if count <= 256 else []
    complete = count <= 256 and len(offsets) == count
    check(canonical(r['offsets']) == canonical(offsets) and canonical(r['excerpts']) == canonical(excerpts), 'incorrect offset/excerpt evidence')
    check(r['coverage_complete'] is complete and r['policy']['mode'] == ('full' if complete else 'limited'), 'false completeness policy')
    claim = ('FOUND' if count else 'PROVEN_EMPTY') if complete else 'UNTRUSTED'
    check(r['status'] == claim and r['reason'] == ('verified-single-corpus' if complete else 'LIMITED_COMPLETENESS'), 'forged status/reason')
    return dict(status=('VERIFIED_SMALL_SCAN' if a.test_fixture else 'VERIFIED_100MB_SCAN') if complete else 'UNTRUSTED',
        claim=claim, reason='LIMITED_COMPLETENESS' if not complete else 'authenticated-complete-source-scan',
        independent_count=count, count_verified=True, full_source_authenticated=True, scan_bytes=scan['scan_bytes'],
        eof=scan['eof'], chunk_bytes=a.chunk_bytes, stored_oracle_offsets=len(scan['offsets']),
        oracle_offsets_truncated=scan['truncated'], returned_offsets=len(offsets), coverage_complete=complete,
        source_hash_passes=2,
        retained_source_archive_bytes=0, independent_index_source_semantics=False, physical_structure=geometry)


def main():
    p = argparse.ArgumentParser()
    for key in ('manifest', 'root-sha256', 'rlb2', 'rlr2', 'ef', 'decoder', 'materialized', 'receipt',
                'pattern-hex', 'research', 'reference'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--chunk-bytes', type=int, default=1024**2)
    p.add_argument('--test-fixture', action='store_true')
    a = p.parse_args()
    w = Work()
    started, cpu = time.monotonic(), time.process_time()
    try:
        result = verify(a, w)
    except Exception as e:
        result = dict(status='UNTRUSTED', reason=str(e)[:240])
    result.update(source_read_bytes=w.source_read_bytes, component_read_bytes=w.component_read_bytes,
        sha256_operations=w.sha_operations, hashed_bytes=w.hashed_bytes, wall_seconds=time.monotonic()-started,
        cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    code = 3 if result['status'] == 'UNTRUSTED' else 0
    print(json.dumps(result, sort_keys=True), file=sys.stderr if code else sys.stdout)
    return code


if __name__ == '__main__':
    sys.exit(main())
