#!/usr/bin/env python3
"""Independent composition replay. Stdlib only, NO composition-producer import.

An unchanged independent L9 leaf verifier is invoked as a separate executable.
Full per-document source scan and identities/coverage/lineage are recomputed
here, never inferred from stored success flags. No Merkle/signature claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

PROFILE = 'GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0'
HIST = '81628f843dc8aa633c65594a66d2386bfc3cb5c2'
REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
RESEARCH = ('experiments/core_l10_composition_v0/producer.py', 'experiments/core_l10_composition_v0/verifier.py',
    'experiments/core_l9_exact_query_v0/producer.py', 'experiments/core_l9_exact_query_v0/verifier.py',
    'experiments/loc_text_ef_v0/trusted_materialize_v0.py', 'experiments/loc_text_ef_v0/query_ef_research.py',
    'experiments/loc_text_ef_v0/self_index_exact_range_extractor.py')
FROZEN = ('tools/rlbwt_query_v2.py', 'tools/rlbwt_container_v2.py', 'tools/rlbwt_rank_v2.py')


def check(ok, why):
    if not ok:
        raise ValueError(why)


def canon(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode('ascii')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fields(x, names):
    check(type(x) is dict and set(x) == set(names), 'unknown/missing fields')


def uint(x, maximum=2**64-1):
    return type(x) is int and 0 <= x <= maximum


def hash_string(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def pairs(items):
    d = {}
    for k, v in items:
        check(k not in d, 'duplicate JSON key')
        d[k] = v
    return d


def parsed(raw):
    x = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    check(type(x) is dict and canon(x) == raw, 'noncanonical JSON')
    return x


class Work:
    def __init__(self, research, reference):
        self.research, self.reference = research, reference
        self.read_bytes = self.sha_operations = self.source_bytes = self.blocks = self.leaves = 0
        self.leaf_processes = self.leaf_source_bytes = self.leaf_sha_operations = 0

    def read(self, path, cap):
        path = Path(path)
        check(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, 'missing/nonregular/oversize '+str(path))
        raw = path.read_bytes()
        check(len(raw) <= cap, 'file grew')
        self.read_bytes += len(raw)
        return raw

    def hash(self, raw):
        self.sha_operations += 1
        return digest(raw)

    def obj(self, path, record):
        fields(record, ('bytes', 'sha256'))
        check(uint(record['bytes'], 1024**2) and record['bytes'] > 0 and hash_string(record['sha256']), 'bad object identity')
        raw = self.read(path, 1024**2)
        check(len(raw) == record['bytes'] and self.hash(raw) == record['sha256'], 'physical object identity mismatch')
        return raw

    def codes(self):
        check(subprocess.check_output(['git', '-C', str(self.reference), 'rev-parse', 'HEAD'], text=True).strip() == REF, 'frozen reference HEAD')
        records = {}
        for label, directory, names in [('research', self.research, RESEARCH), ('reference', self.reference, FROZEN)]:
            for name in names:
                data = self.read(directory/name, 1024**2)
                if label == 'reference':
                    check(data == subprocess.check_output(['git', '-C', str(directory), 'show', REF+':'+name]), 'modified frozen code')
                records[label+'/'+name] = dict(bytes=len(data), sha256=self.hash(data))
        return records

    def identities(self, docs):
        preimage = b'GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\0'+len(docs).to_bytes(8, 'big')
        source = b'GLYPH_OPERATOR_CORPUS_MANIFEST_V1\0'+len(docs).to_bytes(8, 'big')
        paths = []
        for i, d in enumerate(docs):
            fields(d, ('doc_id', 'relative_path_bytes_hex', 'byte_length', 'sha256'))
            check(type(d['doc_id']) is int and d['doc_id'] == i and uint(d['byte_length'], 256) and hash_string(d['sha256']), 'doc slot/length/SHA')
            path = bytes.fromhex(d['relative_path_bytes_hex'])
            check(path and path.hex() == d['relative_path_bytes_hex'] and len(path) <= 128
                  and all(32 <= b < 127 and b != 92 for b in path) and not path.startswith(b'/')
                  and all(p not in (b'', b'.', b'..') for p in path.split(b'/')), 'unsafe/noncanonical path')
            paths.append(path)
            tail = d['byte_length'].to_bytes(8, 'big')+bytes.fromhex(d['sha256'])
            preimage += i.to_bytes(8, 'big')+tail
            source += i.to_bytes(8, 'big')+len(path).to_bytes(8, 'big')+path+tail
        check(paths == sorted(set(paths)), 'duplicate/reordered source paths')
        return self.hash(preimage), self.hash(source)

    def snapshot(self, bundle, path, pin, generation, previous_path=None, previous_pin=None):
        data = self.read(path, 65536)
        check(hash_string(pin) and self.hash(data) == pin, 'external root mismatch')
        root = parsed(data)
        fields(root, ('format', 'historical_reference', 'generation', 'documents', 'leaves', 'blocks', 'runtime_corpus_id',
                      'source_manifest_id', 'parent', 'dependencies', 'research_composition_root_id'))
        check(root['format'] == PROFILE and root['historical_reference'] == HIST and uint(generation, 1)
              and type(root['generation']) is int and root['generation'] == generation, 'generation/profile mismatch')
        body = dict(root)
        claimed = body.pop('research_composition_root_id')
        check(self.hash(PROFILE.encode()+b'\0ROOT\0'+canon(body)) == claimed, 'research root identity mismatch')
        docs, leaves, blocks = root['documents'], root['leaves'], root['blocks']
        check(type(docs) is list and 9 <= len(docs) <= 64 and type(leaves) is list and len(leaves) == len(docs)
              and type(blocks) is list and 0 < len(blocks) <= len(docs), 'incomplete document/leaf/block domains')
        cid, sid = self.identities(docs)
        check(root['runtime_corpus_id'] == cid and root['source_manifest_id'] == sid, 'global identity mismatch')
        fields(root['dependencies'], ('code', 'decoder'))
        check(canon(root['dependencies']['code']) == canon(self.codes()), 'dependency code identity mismatch')
        self.obj(bundle/'decoder', root['dependencies']['decoder'])
        bindings = []
        for i, item in enumerate(leaves):
            fields(item, ('doc_id', 'binding'))
            check(type(item['doc_id']) is int and item['doc_id'] == i, 'duplicate/reordered leaf slot')
            folder = bundle/'leaves'/str(i)
            binding = parsed(self.obj(folder/'binding.json', item['binding']))
            self.leaves += 1
            fields(binding, ('format', 'document', 'kind', 'index_root', 'artifacts'))
            check(binding['format'] == PROFILE+'_LEAF' and canon(binding['document']) == canon(docs[i]), 'leaf slot document mismatch')
            if docs[i]['byte_length'] == 0:
                check(binding['kind'] == 'EMPTY_MANIFEST_ONLY' and binding['index_root'] is None and binding['artifacts'] == {}
                      and docs[i]['sha256'] == self.hash(b''), 'invalid explicit empty leaf')
                files = {'binding.json'}
            else:
                check(binding['kind'] == 'RLB2_RLR2_EF', 'nonempty source missing index')
                lr = parsed(self.obj(folder/'root.json', binding['index_root']))
                check(canon(lr.get('source')) == canon(dict(bytes=docs[i]['byte_length'], sha256=docs[i]['sha256']))
                      and canon(lr.get('decoder')) == canon(dict(sha256=root['dependencies']['decoder']['sha256']))
                      and canon(lr.get('artifacts')) == canon(binding['artifacts']), 'leaf root source/component binding')
                fields(binding['artifacts'], ('rlb2', 'rlr2', 'ef'))
                for name in ('rlb2', 'rlr2', 'ef'):
                    self.obj(folder/name, binding['artifacts'][name])
                files = {'binding.json', 'root.json', 'rlb2', 'rlr2', 'ef'}
            check(set(p.name for p in folder.iterdir()) == files, 'unaccounted leaf state')
            bindings.append(binding)
        flattened = []
        next_doc = 0
        for i, block in enumerate(blocks):
            fields(block, ('ordinal', 'first_doc_id', 'document_count', 'manifest'))
            check(type(block['ordinal']) is int and block['ordinal'] == i and type(block['first_doc_id']) is int
                  and block['first_doc_id'] == next_doc and uint(block['document_count'], len(docs)) and block['document_count'] > 0,
                  'block range/order/coverage')
            physical = parsed(self.obj(bundle/'blocks'/(block['manifest']['sha256']+'.json'), block['manifest']))
            self.blocks += 1
            fields(physical, ('format', 'ordinal', 'first_doc_id', 'document_count', 'leaves'))
            expected = dict(format=PROFILE+'_BLOCK', ordinal=i, first_doc_id=next_doc,
                            document_count=block['document_count'], leaves=leaves[next_doc:next_doc+block['document_count']])
            check(canon(physical) == canon(expected), 'block binding does not match root range')
            flattened += physical['leaves']
            next_doc += block['document_count']
        check(next_doc == len(docs) and canon(flattened) == canon(leaves), 'missing/duplicate block coverage')
        if generation == 0:
            check(root['parent'] is None and previous_path is None and previous_pin is None, 'unexpected initial parent')
            lineage = dict(mode='INITIAL', old_root_sha256=None, old_document_count=0, appended_document_count=len(docs))
        else:
            fields(root['parent'], ('sha256', 'research_composition_root_id', 'document_count'))
            check(previous_path is not None and hash_string(previous_pin) and root['parent']['sha256'] == previous_pin,
                  'missing separately expected old root pin')
            old, _ = self.snapshot(bundle, previous_path, previous_pin, 0)
            n = len(old['documents'])
            check(type(root['parent']['document_count']) is int and root['parent']['document_count'] == n
                  and root['parent']['research_composition_root_id'] == old['research_composition_root_id'], 'wrong parent identity')
            check(len(docs) >= n+3 and canon(docs[:n]) == canon(old['documents'])
                  and canon(leaves[:n]) == canon(old['leaves'])
                  and canon(blocks[:len(old['blocks'])]) == canon(old['blocks']), 'old manifest/leaf prefix changed or append absent')
            check(cid != old['runtime_corpus_id'] and sid != old['source_manifest_id'], 'unchanged append identity')
            lineage = dict(mode='ORDERED_MANIFEST_PREFIX_REPLAY_V0', old_root_sha256=previous_pin,
                           old_document_count=n, appended_document_count=len(docs)-n)
        return root, lineage


def verify(a, w):
    root, lineage = w.snapshot(a.bundle, a.root, a.pin, a.generation, a.previous_root, a.previous_pin)
    r = parsed(w.read(a.receipt, 512*1024))
    fields(r, ('format', 'root_sha256', 'research_composition_root_id', 'generation', 'runtime_corpus_id', 'source_manifest_id',
        'query_hex', 'query_length_bytes', 'query_sha256', 'max_offsets', 'match_count', 'returned_count', 'coordinates',
        'excerpts', 'bounded', 'offsets_complete', 'coverage_complete', 'coverage', 'block_counts', 'leaf_results',
        'lineage', 'status', 'reason'))
    pattern = bytes.fromhex(a.query_hex)
    check(pattern and len(pattern) <= 4097 and pattern.hex() == a.query_hex and uint(a.limit, 256), 'expected query/policy')
    blocks = [b['manifest']['sha256'] for b in root['blocks']]
    leaves = [l['binding']['sha256'] for l in root['leaves']]
    coverage = dict(expected_blocks=blocks, verified_blocks=blocks, queried_blocks=blocks,
                    expected_leaves=leaves, verified_leaves=leaves, queried_leaves=leaves)
    check(canon(r['coverage']) == canon(coverage) and r['coverage_complete'] is True, 'false complete block/leaf coverage')
    check(type(r['leaf_results']) is list and len(r['leaf_results']) == len(leaves), 'unacknowledged/missing leaf result')
    count, coordinates, excerpts, counts = 0, [], [], []
    with tempfile.TemporaryDirectory(prefix='l10-independent-leaf-') as temporary:
        receipt_file = Path(temporary)/'leaf.json'
        for i, (doc, result) in enumerate(zip(root['documents'], r['leaf_results'])):
            fields(result, ('doc_id', 'binding_sha256', 'count', 'receipt'))
            check(type(result['doc_id']) is int and result['doc_id'] == i and result['binding_sha256'] == leaves[i], 'leaf result slot mismatch')
            data = w.read(a.oracles/(str(i)+'.raw'), 256)
            w.source_bytes += len(data)
            check(len(data) == doc['byte_length'] and w.hash(data) == doc['sha256'], 'source oracle identity mismatch')
            offsets = [j for j in range(len(data)) if data.startswith(pattern, j)]
            check(type(result['count']) is int and result['count'] == len(offsets), 'leaf count mismatch')
            if not data:
                check(result['receipt'] is None and doc['sha256'] == digest(b''), 'empty leaf has fake index result')
            else:
                leaf = a.bundle/'leaves'/str(i)
                binding = parsed(w.read(leaf/'binding.json', 65536))
                receipt_file.write_bytes(canon(result['receipt']))
                args = [sys.executable, '-I', str(a.research/'experiments/core_l9_exact_query_v0/verifier.py'),
                    '--manifest', str(leaf/'root.json'), '--root-sha256', binding['index_root']['sha256'],
                    '--rlb2', str(leaf/'rlb2'), '--rlr2', str(leaf/'rlr2'), '--ef', str(leaf/'ef'),
                    '--decoder', str(a.bundle/'decoder'), '--materialized', str(a.oracles/(str(i)+'.raw')),
                    '--receipt', str(receipt_file), '--pattern-hex', a.query_hex, '--limit', '256',
                    '--research', str(a.research), '--reference', str(a.reference)]
                child = subprocess.run(args, capture_output=True, text=True, timeout=30)
                check(child.returncode == 0, 'independent leaf verifier rejected '+str(i)+': '+child.stderr[-200:])
                info = json.loads(child.stdout)
                check(info['status'] == 'VERIFIED_SMALL', 'leaf verifier did not affirm actual replay')
                w.leaf_processes += 1
                w.leaf_source_bytes += info['source_read_bytes']
                w.leaf_sha_operations += info['sha256_operations']
            counts.append(len(offsets))
            count += len(offsets)
            for offset in offsets:
                if len(coordinates) < a.limit:
                    coordinates.append([i, offset])
                    excerpts.append(dict(doc_id=i, doc_offset=offset, byte_length=len(pattern),
                        sha256=w.hash(data[offset:offset+len(pattern)]), source_sha256=doc['sha256'], bytes_included=False))
    complete = len(coordinates) == count
    expected = dict(format=PROFILE+'_RECEIPT', root_sha256=a.pin, research_composition_root_id=root['research_composition_root_id'],
        generation=a.generation, runtime_corpus_id=root['runtime_corpus_id'], source_manifest_id=root['source_manifest_id'],
        query_hex=a.query_hex, query_length_bytes=len(pattern), query_sha256=w.hash(pattern), max_offsets=a.limit,
        match_count=count, returned_count=len(coordinates), coordinates=coordinates, excerpts=excerpts,
        bounded=not complete, offsets_complete=complete, coverage_complete=True, coverage=coverage,
        block_counts=[dict(ordinal=b['ordinal'], count=sum(counts[b['first_doc_id']:b['first_doc_id']+b['document_count']])) for b in root['blocks']],
        lineage=lineage, status=('FOUND' if count else 'PROVEN_EMPTY') if complete else 'UNTRUSTED',
        reason='verified-single-snapshot' if complete else 'LIMITED_COMPLETENESS')
    for name, value in expected.items():
        check(canon(r[name]) == canon(value), 'receipt disagreement: '+name)
    return dict(status='VERIFIED_APPEND_RESEARCH' if complete else 'UNTRUSTED', claim=expected['status'],
                reason=expected['reason'], count_verified=True, match_count=count, coordinates=coordinates,
                generation=a.generation, lineage=lineage, full_leaf_block_coverage=True, source_scan_bytes=w.source_bytes)


def main():
    p = argparse.ArgumentParser()
    for key in ('bundle', 'root', 'oracles', 'receipt', 'research', 'reference'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('pin', 'query-hex'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--generation', type=int, required=True)
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--previous-root', type=Path)
    p.add_argument('--previous-pin')
    a = p.parse_args()
    w = Work(a.research, a.reference)
    started, cpu = time.monotonic(), time.process_time()
    try:
        result = verify(a, w)
    except Exception as e:
        result = dict(status='UNTRUSTED', reason=str(e)[:350])
    result.update(read_bytes=w.read_bytes, sha256_operations=w.sha_operations, root_block_records=w.blocks,
        leaf_bindings=w.leaves, leaf_verifier_processes=w.leaf_processes, leaf_source_read_bytes=w.leaf_source_bytes,
        leaf_sha256_operations=w.leaf_sha_operations, wall_seconds=time.monotonic()-started,
        cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        leaf_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    code = 3 if result['status'] == 'UNTRUSTED' else 0
    print(json.dumps(result, sort_keys=True), file=sys.stderr if code else sys.stdout)
    return code


if __name__ == '__main__':
    sys.exit(main())
