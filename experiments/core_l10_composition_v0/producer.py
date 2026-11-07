#!/usr/bin/env python3
"""Non-normative full-coverage leaf fanout, not an operator/Index Forest ABI."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILE = 'GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0'
HIST = '81628f843dc8aa633c65594a66d2386bfc3cb5c2'
REF = 'c33cb493bc75f13ee4a904fabb190a29a196c94b'
CODE = ('experiments/core_l10_composition_v0/producer.py', 'experiments/core_l10_composition_v0/verifier.py',
        'experiments/core_l9_exact_query_v0/producer.py', 'experiments/core_l9_exact_query_v0/verifier.py',
        'experiments/loc_text_ef_v0/trusted_materialize_v0.py', 'experiments/loc_text_ef_v0/query_ef_research.py',
        'experiments/loc_text_ef_v0/self_index_exact_range_extractor.py')
FROZEN_CODE = ('tools/rlbwt_query_v2.py', 'tools/rlbwt_container_v2.py', 'tools/rlbwt_rank_v2.py')
ROOT_KEYS = ('format', 'historical_reference', 'generation', 'documents', 'leaves', 'blocks',
             'runtime_corpus_id', 'source_manifest_id', 'parent', 'dependencies', 'research_composition_root_id')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode('ascii')


def sha(x):
    return hashlib.sha256(x).hexdigest()


def integer(x, high=2**64-1):
    return type(x) is int and 0 <= x <= high


def keys(x, names):
    require(type(x) is dict and set(x) == set(names), 'unknown/missing fields')


def digest_string(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def read(path, cap=65536):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, 'missing/nonregular/oversize: '+str(path))
    data = path.read_bytes()
    require(len(data) <= cap, 'file grew')
    return data


def decode(raw):
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(type(value) is dict and canonical(value) == raw, 'noncanonical JSON')
    return value


def identity(path):
    data = read(path, 1024**2)
    return dict(bytes=len(data), sha256=sha(data))


def object_bytes(path, record):
    keys(record, ('bytes', 'sha256'))
    require(integer(record['bytes'], 1024**2) and record['bytes'] > 0 and digest_string(record['sha256']), 'object identity type')
    raw = read(path, 1024**2)
    require(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'physical identity mismatch: '+str(path))
    return raw


def global_ids(documents):
    corpus = hashlib.sha256(b'GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\0'+len(documents).to_bytes(8, 'big'))
    source = hashlib.sha256(b'GLYPH_OPERATOR_CORPUS_MANIFEST_V1\0'+len(documents).to_bytes(8, 'big'))
    previous = b''
    for i, doc in enumerate(documents):
        keys(doc, ('doc_id', 'relative_path_bytes_hex', 'byte_length', 'sha256'))
        require(integer(doc['doc_id']) and doc['doc_id'] == i and integer(doc['byte_length'], 256)
                and digest_string(doc['sha256']), 'document ID/length/SHA')
        path = bytes.fromhex(doc['relative_path_bytes_hex'])
        require(path.hex() == doc['relative_path_bytes_hex'] and 0 < len(path) <= 128 and path > previous
                and all(32 <= b < 127 and b != 92 for b in path)
                and not path.startswith(b'/') and all(p not in (b'', b'.', b'..') for p in path.split(b'/')), 'canonical path subset')
        previous = path
        suffix = doc['byte_length'].to_bytes(8, 'big')+bytes.fromhex(doc['sha256'])
        corpus.update(i.to_bytes(8, 'big')+suffix)
        source.update(i.to_bytes(8, 'big')+len(path).to_bytes(8, 'big')+path+suffix)
    return corpus.hexdigest(), source.hexdigest()


def root_id(root):
    body = dict(root)
    body.pop('research_composition_root_id', None)
    return sha(PROFILE.encode()+b'\0ROOT\0'+canonical(body))


def code_identity(reference):
    require(subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip() == REF, 'reference HEAD')
    files = {}
    for prefix, folder, names in [('research', REPO, CODE), ('reference', reference, FROZEN_CODE)]:
        for name in names:
            raw = read(folder/name, 1024**2)
            if prefix == 'reference':
                require(raw == subprocess.check_output(['git', '-C', str(folder), 'show', REF+':'+name]), 'frozen code changed')
            files[prefix+'/'+name] = dict(bytes=len(raw), sha256=sha(raw))
    return files


def load_root(bundle, path, pin, generation, reference, previous_path=None, previous_pin=None):
    raw = read(path)
    require(digest_string(pin) and sha(raw) == pin, 'wrong externally expected root')
    root = decode(raw)
    keys(root, ROOT_KEYS)
    require(root['format'] == PROFILE and root['historical_reference'] == HIST and integer(generation, 1)
            and type(root['generation']) is int and root['generation'] == generation, 'snapshot generation/profile')
    require(root_id(root) == root['research_composition_root_id'], 'research layout root identity')
    docs, leaves, blocks = root['documents'], root['leaves'], root['blocks']
    require(type(docs) is list and 9 <= len(docs) <= 64 and type(leaves) is list and len(leaves) == len(docs)
            and type(blocks) is list and 1 <= len(blocks) <= len(docs), 'document/leaf/block coverage')
    corpus, source = global_ids(docs)
    require(root['runtime_corpus_id'] == corpus and root['source_manifest_id'] == source, 'ordered document identities')
    keys(root['dependencies'], ('decoder', 'code'))
    object_bytes(bundle/'decoder', root['dependencies']['decoder'])
    require(canonical(root['dependencies']['code']) == canonical(code_identity(reference)), 'code dependency identity')
    bindings = []
    for i, (doc, entry) in enumerate(zip(docs, leaves)):
        keys(entry, ('doc_id', 'binding'))
        require(type(entry['doc_id']) is int and entry['doc_id'] == i, 'leaf slot changed/duplicated')
        directory = bundle/'leaves'/str(i)
        binding = decode(object_bytes(directory/'binding.json', entry['binding']))
        keys(binding, ('format', 'document', 'kind', 'index_root', 'artifacts'))
        require(binding['format'] == PROFILE+'_LEAF' and canonical(binding['document']) == canonical(doc), 'leaf/document identity')
        if doc['byte_length'] == 0:
            require(binding['kind'] == 'EMPTY_MANIFEST_ONLY' and binding['index_root'] is None and binding['artifacts'] == {}
                    and doc['sha256'] == sha(b''), 'explicit zero-length identity')
            expected_files = {'binding.json'}
        else:
            require(binding['kind'] == 'RLB2_RLR2_EF', 'nonempty leaf cannot be manifest-only')
            lr = decode(object_bytes(directory/'root.json', binding['index_root']))
            keys(binding['artifacts'], ('rlb2', 'rlr2', 'ef'))
            require(lr.get('source') == dict(bytes=doc['byte_length'], sha256=doc['sha256'])
                    and canonical(lr.get('artifacts')) == canonical(binding['artifacts'])
                    and lr.get('decoder') == dict(sha256=root['dependencies']['decoder']['sha256']), 'leaf root/source binding')
            for name in ('rlb2', 'rlr2', 'ef'):
                object_bytes(directory/name, binding['artifacts'][name])
            expected_files = {'binding.json', 'root.json', 'rlb2', 'rlr2', 'ef'}
        require(set(p.name for p in directory.iterdir()) == expected_files, 'unaccounted leaf payload')
        bindings.append(binding)
    base = 0
    for i, block in enumerate(blocks):
        keys(block, ('ordinal', 'first_doc_id', 'document_count', 'manifest'))
        require(type(block['ordinal']) is int and block['ordinal'] == i and type(block['first_doc_id']) is int
                and block['first_doc_id'] == base and integer(block['document_count'], len(docs)) and block['document_count'] > 0,
                'contiguous block range/order')
        end = base+block['document_count']
        expected = dict(format=PROFILE+'_BLOCK', ordinal=i, first_doc_id=base,
                        document_count=block['document_count'], leaves=leaves[base:end])
        actual = decode(object_bytes(bundle/'blocks'/(block['manifest']['sha256']+'.json'), block['manifest']))
        require(canonical(actual) == canonical(expected), 'physical block membership mismatch')
        base = end
    require(base == len(docs), 'incomplete block coverage')
    if generation == 0:
        require(root['parent'] is None and previous_path is None and previous_pin is None, 'initial root has parent')
        lineage = dict(mode='INITIAL', old_root_sha256=None, old_document_count=0, appended_document_count=len(docs))
    else:
        keys(root['parent'], ('sha256', 'research_composition_root_id', 'document_count'))
        require(previous_path is not None and digest_string(previous_pin) and root['parent']['sha256'] == previous_pin,
                'separately pinned expected V0 required')
        old, _bindings, _lineage = load_root(bundle, previous_path, previous_pin, 0, reference)
        n = len(old['documents'])
        require(type(root['parent']['document_count']) is int and root['parent']['document_count'] == n
                and root['parent']['research_composition_root_id'] == old['research_composition_root_id'], 'old root mismatch')
        require(len(docs) >= n+3 and canonical(docs[:n]) == canonical(old['documents'])
                and canonical(leaves[:n]) == canonical(old['leaves'])
                and canonical(blocks[:len(old['blocks'])]) == canonical(old['blocks']), 'append prefix/old leaf changed')
        require(corpus != old['runtime_corpus_id'] and source != old['source_manifest_id'], 'append did not change corpus')
        lineage = dict(mode='ORDERED_MANIFEST_PREFIX_REPLAY_V0', old_root_sha256=previous_pin,
                       old_document_count=n, appended_document_count=len(docs)-n)
    return root, bindings, lineage


class RootView:
    def __init__(self, bundle, path, pin, generation, reference, previous_path=None, previous_pin=None):
        started = time.monotonic()
        self.states = []
        self.bundle, self.pin = Path(bundle), pin
        self.root, self.bindings, self.lineage = load_root(self.bundle, path, pin, generation, Path(reference), previous_path, previous_pin)
        spec = importlib.util.spec_from_file_location('unchanged_l9_leaf_producer', REPO/'experiments/core_l9_exact_query_v0/producer.py')
        primitive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(primitive)
        try:
            for i, binding in enumerate(self.bindings):
                directory = self.bundle/'leaves'/str(i)
                self.states.append(None if binding['kind'] == 'EMPTY_MANIFEST_ONLY' else primitive.QueryState(
                    directory/'root.json', binding['index_root']['sha256'], directory/'rlb2', directory/'rlr2',
                    directory/'ef', self.bundle/'decoder', Path(reference)))
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.monotonic()-started

    def close(self):
        for state in self.states:
            if state is not None:
                state.close()

    def query(self, pattern, limit):
        require(type(pattern) is bytes and 0 < len(pattern) <= 4097 and integer(limit, 256), 'query/global limit admission')
        started, cpu = time.monotonic(), time.process_time()
        total, coordinates, excerpts, leaf_results, block_counts = 0, [], [], [], []
        for block in self.root['blocks']:
            count = 0
            for i in range(block['first_doc_id'], block['first_doc_id']+block['document_count']):
                state = self.states[i]
                primitive = None if state is None else state.query(pattern, 256)
                number = 0 if primitive is None else primitive['count']
                require(primitive is None or primitive['coverage_complete'] is True, 'leaf completeness budget')
                require(total+number <= 2**64-1, 'count overflow')
                total += number
                count += number
                leaf_results.append(dict(doc_id=i, binding_sha256=self.root['leaves'][i]['binding']['sha256'],
                                         count=number, receipt=primitive))
                if primitive:
                    for off, fragment in zip(primitive['offsets'], primitive['excerpts']):
                        if len(coordinates) < limit:
                            coordinates.append([i, off])
                            excerpts.append(dict(doc_id=i, doc_offset=off, byte_length=len(pattern),
                                sha256=fragment['sha256'], source_sha256=self.root['documents'][i]['sha256'], bytes_included=False))
            block_counts.append(dict(ordinal=block['ordinal'], count=count))
        expected_blocks = [b['manifest']['sha256'] for b in self.root['blocks']]
        expected_leaves = [l['binding']['sha256'] for l in self.root['leaves']]
        complete = len(coordinates) == total
        r = dict(format=PROFILE+'_RECEIPT', root_sha256=self.pin,
            research_composition_root_id=self.root['research_composition_root_id'], generation=self.root['generation'],
            runtime_corpus_id=self.root['runtime_corpus_id'], source_manifest_id=self.root['source_manifest_id'],
            query_hex=pattern.hex(), query_length_bytes=len(pattern), query_sha256=sha(pattern), max_offsets=limit,
            match_count=total, returned_count=len(coordinates), coordinates=coordinates, excerpts=excerpts,
            bounded=not complete, offsets_complete=complete, coverage_complete=True,
            coverage=dict(expected_blocks=expected_blocks, verified_blocks=expected_blocks, queried_blocks=expected_blocks,
                          expected_leaves=expected_leaves, verified_leaves=expected_leaves, queried_leaves=expected_leaves),
            block_counts=block_counts, leaf_results=leaf_results, lineage=self.lineage,
            status=('FOUND' if total else 'PROVEN_EMPTY') if complete else 'UNTRUSTED',
            reason='verified-single-snapshot' if complete else 'LIMITED_COMPLETENESS')
        require(len(canonical(r)) <= 512*1024, 'bounded receipt budget')
        self.metrics = dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return r


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'root', 'pin', 'reference'):
        p.add_argument('--'+name, required=True, type=Path if name != 'pin' else str)
    p.add_argument('--generation', type=int, required=True)
    p.add_argument('--previous-root', type=Path)
    p.add_argument('--previous-pin')
    p.add_argument('--query-hex')
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--queries', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    view = None
    try:
        view = RootView(a.bundle, a.root, a.pin, a.generation, a.reference, a.previous_root, a.previous_pin)
        if a.queries:
            require(a.output is not None, 'batch output required')
            a.output.mkdir(exist_ok=False)
            cases = []
            for case in json.loads(read(a.queries)):
                require(type(case['name']) is str and case['name'].isalnum(), 'safe case name')
                r = view.query(bytes.fromhex(case['hex']), case['limit'])
                metrics = dict(view.metrics)
                require(canonical(r) == canonical(view.query(bytes.fromhex(case['hex']), case['limit'])), 'nondeterministic receipt')
                (a.output/(case['name']+'.json')).write_bytes(canonical(r))
                cases.append(dict(name=case['name'], status=r['status'], count=r['match_count'], metrics=metrics,
                                  receipt_bytes=len(canonical(r))))
            print(json.dumps(dict(startup_seconds=view.startup_seconds, cases=cases), sort_keys=True))
            return 0
        pattern = bytes.fromhex(a.query_hex)
        require(pattern.hex() == a.query_hex, 'canonical query hex')
        r = view.query(pattern, a.limit)
        sys.stdout.buffer.write(canonical(r))
        return 3 if r['status'] == 'UNTRUSTED' else 0
    except Exception as e:
        print(json.dumps(dict(status='UNTRUSTED', reason=str(e)[:300])), file=sys.stderr)
        return 3
    finally:
        if view:
            view.close()


if __name__ == '__main__':
    sys.exit(main())
