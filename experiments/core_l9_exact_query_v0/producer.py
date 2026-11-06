#!/usr/bin/env python3
"""Bounded single-corpus CORE-L9 research query producer. No raw-source input."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

sys.dont_write_bytecode = True

FROZEN = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
LOC = RESEARCH / 'experiments/loc_text_ef_v0'
sys.path.insert(0, str(LOC))
import trusted_materialize_v0 as l8
from query_ef_research import make_runtime
from self_index_exact_range_extractor import EFByteExtractor

SCHEMA = 'GLYPH_CORE_L9_EXACT_QUERY_RECEIPT_V0'
MAX_SOURCE, MAX_COMPONENT, MAX_RECEIPT, MAX_HITS = 4096, 1024**2, 65536, 256
SCOPE = 'single-complete-corpus; corpus-byte-offsets; no-doc-map; no-currentness'
CODE = ('experiments/core_l9_exact_query_v0/producer.py',
        'experiments/core_l9_exact_query_v0/verifier.py',
        'experiments/loc_text_ef_v0/trusted_materialize_v0.py',
        'experiments/loc_text_ef_v0/query_ef_research.py',
        'experiments/loc_text_ef_v0/self_index_exact_range_extractor.py')
REF_CODE = ('tools/rlbwt_query_v2.py', 'tools/rlbwt_container_v2.py', 'tools/rlbwt_rank_v2.py')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def runtime_identity(reference):
    if subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'],
                               text=True).strip() != FROZEN:
        raise ValueError('wrong frozen reference')
    files = {}
    for prefix, root, paths in [('research', RESEARCH, CODE), ('reference', reference, REF_CODE)]:
        for name in paths:
            raw = (root / name).read_bytes()
            if prefix == 'reference':
                committed = subprocess.check_output(['git', '-C', str(root), 'show', FROZEN + ':' + name])
                if raw != committed:
                    raise ValueError('modified frozen runtime source')
            files[prefix + '/' + name] = dict(bytes=len(raw), sha256=sha(raw))
    return dict(profile='CORE_L9_FROZEN_RANKV2_PHYSICAL_EF_V0', files=files,
                revision_sha256=sha(canonical(files)))


def validate_ef_select(loc):
    """Isolated adapter: validate select checkpoints even for zero-hit queries.

    The existing extractor checks row order/quotient coverage but does not use
    select checkpoints. An empty FM interval otherwise never exercises select.
    Keep the prerequisite EF implementation unchanged.
    """
    sample = 0
    for pos in range(loc.highbits):
        if (loc.high[pos//8] >> (pos % 8)) & 1:
            if sample % loc.stride == 0 and struct.unpack_from('<I', loc.cp, 4*(sample//loc.stride))[0] != pos:
                raise ValueError('invalid EF select checkpoint before query')
            sample += 1
    for raw, bits in ((loc.lows, loc.m*loc.low), (loc.high, loc.highbits),
                      (loc.values, loc.m*loc.width)):
        if bits % 8 and raw[-1] >> (bits % 8):
            raise ValueError('noncanonical EF padding')
    if sample != loc.m:
        raise ValueError('invalid EF sample coverage')


class QueryState:
    def __init__(self, manifest, pin, rlb2, rlr2, ef, decoder, reference):
        self.rt = None
        self.stage = tempfile.TemporaryDirectory(prefix='core-l9-private-')
        try:
            self.runtime = runtime_identity(Path(reference).resolve())
            rootpath = Path(manifest)
            if rootpath.is_symlink() or not rootpath.is_file() or rootpath.stat().st_size > 8192:
                raise ValueError('invalid bounded root')
            raw = rootpath.read_bytes()
            self.root = l8.parse_manifest(raw, pin)
            if raw != canonical(self.root):
                raise ValueError('noncanonical root')
            if self.root['source']['bytes'] > MAX_SOURCE:
                raise ValueError('bounded small-corpus gate only')
            self.pin = pin
            folder = Path(self.stage.name)
            for name, source in [('rlb2', rlb2), ('rlr2', rlr2), ('ef', ef)]:
                record = self.root['artifacts'][name]
                if record['bytes'] > MAX_COMPONENT:
                    raise ValueError('component budget exceeded')
                l8.pinned_input(Path(source), folder / name, record)
            d = Path(decoder)
            if not 0 < d.lstat().st_size <= MAX_COMPONENT:
                raise ValueError('decoder budget exceeded')
            l8.pinned_input(d, folder / 'decoder', dict(bytes=d.lstat().st_size,
                                                     sha256=self.root['decoder']['sha256']))
            # Reuse the frozen full checksum + deterministic rank rebuild check;
            # RankV2's query constructor alone is not a complete file validator.
            valid = subprocess.run([sys.executable, str(Path(reference)/'tools/rlbwt_rank_v2.py'),
                                    'verify', str(folder/'rlb2'), str(folder/'rlr2')],
                                   capture_output=True, text=True, timeout=30)
            if valid.returncode:
                raise ValueError('frozen rank/container validation failed: '+valid.stderr[:160])
            self.rt = make_runtime(Path(reference) / 'tools', folder/'rlb2', folder/'rlr2', folder/'ef')
            # make_runtime imports the frozen implementation by module name.
            if Path(sys.modules['rlbwt_query_v2'].__file__).resolve() != (Path(reference)/'tools/rlbwt_query_v2.py').resolve():
                raise ValueError('unexpected runtime import origin')
            self.extractor = EFByteExtractor(self.rt)
            validate_ef_select(self.rt.locate)
            if self.rt.corpus_bytes != self.root['source']['bytes']:
                raise ValueError('source/runtime geometry mismatch')
            # Verify decoded-BWT historical checksum fields with the original
            # decoder, then remove the transient BWT BEFORE any search. It is
            # neither an input to FM/EF search nor a retained source archive.
            check_bwt = folder/'validation.bwt'
            decoded = subprocess.run([sys.executable, str(Path(reference)/'tools/rlbwt_container_v2.py'),
                                      'decode', str(folder/'rlb2'), str(check_bwt)],
                                     capture_output=True, text=True, timeout=30)
            if decoded.returncode:
                raise ValueError('frozen complete RLB2 validation failed: '+decoded.stderr[:160])
            check_bwt.unlink()
            # Entire source is transiently reconstructed from retained index, never an input.
            digest = hashlib.sha256()
            for offset in range(0, self.rt.corpus_bytes, 256):
                data, _ = self.extractor.extract(offset, min(256, self.rt.corpus_bytes-offset))
                digest.update(data)
            if digest.hexdigest() != self.root['source']['sha256']:
                raise ValueError('reconstructed source identity mismatch')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.rt is not None:
            self.rt.close()
            self.rt = None
        self.stage.cleanup()

    def query(self, pattern, limit=MAX_HITS):
        if not isinstance(pattern, bytes) or not 0 < len(pattern) <= MAX_SOURCE+1:
            raise ValueError('nonempty bounded byte query required')
        if type(limit) is not int or not 0 <= limit <= MAX_HITS:
            raise ValueError('invalid canonical offset limit')
        left, right = self.rt.rank.backward_search(pattern)
        count = right-left
        # Never enumerate an unbounded FM interval. Capped output is NOT full evidence.
        result = self.rt.query(pattern, -1 if count <= MAX_HITS else 0)
        if result['count'] != count:
            raise ValueError('index count disagreement')
        all_offsets = result['locate_offsets']
        if len(set(all_offsets)) != len(all_offsets) or all_offsets != sorted(all_offsets):
            raise ValueError('noncanonical locate output')
        offsets = all_offsets[:limit]
        complete = count <= MAX_HITS and len(offsets) == count
        excerpts = []
        for off in offsets:
            data, _ = self.extractor.extract(off, len(pattern))
            if data != pattern:
                raise ValueError('index excerpt differs from exact pattern')
            excerpts.append(dict(offset=off, bytes=len(data), sha256=sha(data), bytes_included=False))
        receipt = dict(schema=SCHEMA, trusted_root_sha256=self.pin, source=self.root['source'],
                       artifacts=self.root['artifacts'], decoder=self.root['decoder'], runtime=self.runtime,
                       scope=SCOPE, pattern_hex=pattern.hex(), pattern_bytes=len(pattern), count=count,
                       policy=dict(mode='full' if complete else 'limited', limit=limit,
                                   order='ascending-corpus-byte-offset'),
                       coverage_complete=complete, offsets=offsets, excerpts=excerpts,
                       status=('FOUND' if count else 'PROVEN_EMPTY') if complete else 'UNTRUSTED',
                       reason='verified-single-corpus' if complete else 'LIMITED_COMPLETENESS')
        if len(canonical(receipt)) > MAX_RECEIPT:
            raise ValueError('receipt budget exceeded')
        return receipt


def main():
    p = argparse.ArgumentParser()
    for key in ('manifest', 'root-sha256', 'rlb2', 'rlr2', 'ef', 'decoder', 'reference', 'pattern-hex'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--limit', type=int, default=MAX_HITS)
    a = p.parse_args()
    state = None
    try:
        state = QueryState(a.manifest, a.root_sha256, a.rlb2, a.rlr2, a.ef, a.decoder, a.reference)
        pattern = bytes.fromhex(a.pattern_hex)
        if pattern.hex() != a.pattern_hex:
            raise ValueError('noncanonical pattern hex')
        result = state.query(pattern, a.limit)
        sys.stdout.buffer.write(canonical(result))
        return 0 if result['status'] != 'UNTRUSTED' else 3
    except Exception as e:
        print(json.dumps(dict(status='UNTRUSTED', reason=str(e)[:200]), sort_keys=True), file=sys.stderr)
        return 3
    finally:
        if state is not None:
            state.close()


if __name__ == '__main__':
    sys.exit(main())
