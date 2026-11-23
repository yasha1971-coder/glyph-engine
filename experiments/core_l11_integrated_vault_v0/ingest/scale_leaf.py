# Derived from preserved core_l9_scale_v0/producer.py. Arbitrary source research
# adapter; <=100000000 bytes. Not the historical enwik8 acceptance profile.
#!/usr/bin/env python3
"""CORE-L9-S adapter: original FM/EF queries after native L8 authentication.

No raw source argument. Private native source materialization is removed before
query(), not passed to count, locate or excerpt recovery. Research only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOC = REPO/'experiments/loc_text_ef_v0'
sys.path.insert(0, str(LOC))
import trusted_materialize_v0 as l8
from query_ef_research import make_runtime
from self_index_exact_range_extractor import EFByteExtractor

spec = importlib.util.spec_from_file_location('preserved_l9_adapter', REPO/'experiments/core_l9_exact_query_v0/producer.py')
small = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small)
canonical, sha = small.canonical, small.sha
REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
SOURCE = '2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8'
PINS = {
    'rlb2': (73365105, '4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3'),
    'rlr2': (12844028, 'e76a49db554860bf907aa307a503240d1e71ef0400282c34548a9726a3f22719'),
    'ef': (2844324, 'c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7')}
CODE = ('experiments/core_l9_scale_v0/producer.py', 'experiments/core_l9_scale_v0/verifier.py',
        'experiments/core_l9_exact_query_v0/producer.py',
        'experiments/loc_text_ef_v0/trusted_materialize_v0.py',
        'experiments/loc_text_ef_v0/query_ef_research.py',
        'experiments/loc_text_ef_v0/self_index_exact_range_extractor.py')
FROZEN_CODE = ('tools/rlbwt_query_v2.py', 'tools/rlbwt_container_v2.py')
PROFILE = 'CORE_L9S_STREAM_SCAN_V0'
SCOPE = 'single-complete-corpus; corpus-byte-offsets; no-doc-map; no-currentness'


def demand(ok, message):
    if not ok:
        raise ValueError(message)


def runtime_identity(reference):
    demand(subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip() == REF,
           'wrong frozen reference')
    records = {}
    for prefix, root, paths in [('research', REPO, CODE), ('reference', reference, FROZEN_CODE)]:
        for name in paths:
            raw = (root/name).read_bytes()
            if prefix == 'reference':
                demand(raw == subprocess.check_output(['git', '-C', str(root), 'show', REF+':'+name]), 'dirty frozen runtime')
            records[prefix+'/'+name] = dict(bytes=len(raw), sha256=sha(raw))
    return dict(profile=PROFILE, files=records, revision_sha256=sha(canonical(records)))


def rank_geometry(rt):
    """Adapter adds complete checkpoint geometry checks; no source/SA oracle."""
    r = rt.rank
    demand(0 < r.raw_length <= 100000001 and 0 < r.rank_step <= 2**32-1,
           'rank size/step admission')
    expected = (r.raw_length+r.rank_step-1)//r.rank_step+1
    demand(r.checkpoint_count == expected, 'rank checkpoint cardinality')
    previous = [0]*257
    prior_stream = 0
    for i in range(expected):
        pos, stream, offset, counts = r._record(i)
        want = r.raw_length if i == expected-1 else i*r.rank_step
        demand(pos == want and sum(counts) == pos and all(x <= y for x, y in zip(previous, counts)),
               'rank prefix geometry/monotonicity')
        demand(prior_stream <= stream <= r.rlb2_payload_bytes and offset <= pos,
               'rank run pointer geometry')
        if i == 0:
            demand(stream == offset == 0, 'rank first checkpoint')
        if i == expected-1:
            demand(stream == r.rlb2_payload_bytes and offset == 0 and counts[256] == 1, 'rank final checkpoint')
        previous, prior_stream = counts, stream


class QueryState:
    def __init__(self, manifest, pin, rlb2, rlr2, ef, decoder, reference, test_fixture=False):
        self.rt = None
        self.stage = tempfile.TemporaryDirectory(prefix='l9s-private-')
        started, cpu = time.monotonic(), time.process_time()
        try:
            self.runtime = runtime_identity(Path(reference).resolve())
            path = Path(manifest)
            demand(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8192, 'bounded regular root required')
            raw = path.read_bytes()
            self.root = l8.parse_manifest(raw, pin)
            demand(raw == canonical(self.root) and 0 < self.root['source']['bytes'] <= 100000000, 'canonical bounded source root')
            self.pin = pin
            stage = Path(self.stage.name)
            for key, src in [('rlb2', rlb2), ('rlr2', rlr2), ('ef', ef)]:
                record = self.root['artifacts'][key]
                demand(record['bytes'] <= 128*1024**2, 'index component admission')
                l8.pinned_input(Path(src), stage/key, record)
            decoder = Path(decoder)
            demand(0 < decoder.lstat().st_size <= 1024**2, 'decoder admission')
            l8.pinned_input(decoder, stage/'decoder', dict(bytes=decoder.lstat().st_size, sha256=self.root['decoder']['sha256']))
            (stage/'decoder').chmod(0o700)
            self.rt = make_runtime(Path(reference)/'tools', stage/'rlb2', stage/'rlr2', stage/'ef')
            demand(Path(sys.modules['rlbwt_query_v2'].__file__).resolve() == (Path(reference)/'tools/rlbwt_query_v2.py').resolve(),
                   'wrong frozen import origin')
            demand(self.rt.corpus_bytes == self.root['source']['bytes'], 'source/index geometry')
            rank_geometry(self.rt)
            self.extractor = EFByteExtractor(self.rt)
            small.validate_ef_select(self.rt.locate)
            demand(self.root['source']['bytes'] <= 100000000, 'scaled research source admission')
            # Existing native streamed restorer authenticates all source bytes.
            # It snapshots mandatory state again: this staging cost is real.
            root_copy = stage/'root.json'
            root_copy.write_bytes(raw)
            oracle = stage/'ephemeral-authenticated-source'
            attestation = l8.materialize(SimpleNamespace(manifest=root_copy, root_sha256=pin,
                rlb2=stage/'rlb2', rlr2=stage/'rlr2', ef=stage/'ef', decoder=stage/'decoder',
                output=oracle, threads=4, mode='rank'))
            demand(attestation['verified_source_before_publication'], 'native source authentication failed')
            oracle.unlink()
            demand(not oracle.exists(), 'ephemeral source retained before query')
            self.startup = dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                native_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                source_authentication=attestation, native_source_bytes=self.rt.corpus_bytes,
                private_snapshot_bytes=sum((stage/n).stat().st_size for n in ('rlb2', 'rlr2', 'ef', 'decoder')),
                l8_nested_snapshot_bytes=sum((stage/n).stat().st_size for n in ('rlb2', 'rlr2', 'ef', 'decoder')),
                ephemeral_source_disk_bytes=self.rt.corpus_bytes,
                sample_map_ram_bytes=self.extractor.sample_rows_ram_bytes)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.rt is not None:
            self.rt.close()
            self.rt = None
        self.stage.cleanup()

    def query(self, pattern, limit=256):
        demand(type(pattern) is bytes and 0 < len(pattern) <= 4097, 'nonempty bounded byte query')
        demand(type(limit) is int and 0 <= limit <= 256, 'limit admission')
        started, cpu = time.monotonic(), time.process_time()
        left, right = self.rt.rank.backward_search(pattern)
        count = right-left
        result = self.rt.query(pattern, -1 if count <= 256 else 0)
        demand(result['count'] == count, 'FM count disagreement')
        all_offsets = result['locate_offsets']
        demand(all_offsets == sorted(set(all_offsets)), 'noncanonical offsets')
        offsets = all_offsets[:limit]
        complete = count <= 256 and len(offsets) == count
        search_wall = time.monotonic()-started
        excerpt_started = time.monotonic()
        excerpts = []
        for offset in offsets:
            raw, _ = self.extractor.extract(offset, len(pattern))
            demand(raw == pattern, 'physical excerpt differs from query')
            excerpts.append(dict(offset=offset, bytes=len(raw), sha256=sha(raw), bytes_included=False))
        r = dict(schema='GLYPH_CORE_L9_EXACT_QUERY_RECEIPT_V0', trusted_root_sha256=self.pin,
            source=self.root['source'], artifacts=self.root['artifacts'], decoder=self.root['decoder'],
            runtime=self.runtime, scope=SCOPE, pattern_hex=pattern.hex(), pattern_bytes=len(pattern), count=count,
            policy=dict(mode='full' if complete else 'limited', limit=limit, order='ascending-corpus-byte-offset'),
            coverage_complete=complete, offsets=offsets, excerpts=excerpts,
            status=('FOUND' if count else 'PROVEN_EMPTY') if complete else 'UNTRUSTED',
            reason='verified-single-corpus' if complete else 'LIMITED_COMPLETENESS')
        demand(len(canonical(r)) <= 65536, 'receipt byte budget')
        self.last_metrics = dict(count_locate_excerpt_wall_seconds=time.monotonic()-started,
            count_locate_wall_seconds=search_wall, excerpt_wall_seconds=time.monotonic()-excerpt_started,
            cpu_seconds=time.process_time()-cpu, count=count, returned_offsets=len(offsets),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return r


def main():
    p = argparse.ArgumentParser()
    for key in ('manifest', 'root-sha256', 'rlb2', 'rlr2', 'ef', 'decoder', 'reference', 'pattern-hex'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--test-fixture', action='store_true', help='explicit <=4096-byte synthetic gate only')
    p.add_argument('--limit', type=int, default=256)
    a = p.parse_args()
    state = None
    try:
        pattern = bytes.fromhex(a.pattern_hex)
        demand(pattern.hex() == a.pattern_hex, 'canonical hex required')
        state = QueryState(a.manifest, a.root_sha256, a.rlb2, a.rlr2, a.ef, a.decoder, a.reference, a.test_fixture)
        r = state.query(pattern, a.limit)
        sys.stdout.buffer.write(canonical(r))
        return 3 if r['status'] == 'UNTRUSTED' else 0
    except Exception as e:
        print(json.dumps(dict(status='UNTRUSTED', reason=str(e)[:240])), file=sys.stderr)
        return 3
    finally:
        if state is not None:
            state.close()


if __name__ == '__main__':
    sys.exit(main())
