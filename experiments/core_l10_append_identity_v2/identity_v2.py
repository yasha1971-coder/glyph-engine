#!/usr/bin/env python3
"""GLYPH CORE-L10 Append Identity V2 -- NONNORMATIVE research producer.

A metadata/identity fixture over authenticated immutable byte objects, *not*
a new RLB2 source store or compatible Composition V1 operator. The original
RLB2/EF multi-document bridge is deliberately an explicitly separate gate.
"""
import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

PROFILE = 'GLYPH_APPEND_IDENTITY_V2_RESEARCH'
DOMAIN = b'GLYPH_APPEND_SNAPSHOT_V2_RESEARCH\x00'
VAULT_ID = '8f' * 16  # deterministic test vault; production must generate it securely
OPS = {'INGEST', 'RENAME', 'REVISE', 'TOMBSTONE'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return (json.dumps(obj, ensure_ascii=True, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def expect(ok, reason):
    if not ok:
        raise ValueError(reason)


def path_hex(path):
    expect(type(path) is bytes and 0 < len(path) <= 4096, 'invalid path length')
    expect(not path.startswith(b'/') and b'\x00' not in path,
           'NUL or absolute path rejected')
    expect(b'\\' not in path and all(part not in (b'', b'.', b'..') for part in path.split(b'/')),
           'unsafe path segment')
    return path.hex()


def derive(events):
    """Strict event replay; keys/fields and history are binding."""
    docs = {}
    paths = {}
    for ordinal, event in enumerate(events):
        expect(type(event) is dict and event.get('ordinal') == ordinal and
               event.get('op') in OPS and type(event.get('doc_id')) is int,
               'invalid event ordinal/op/id')
        did = event['doc_id']
        op = event['op']
        if op == 'INGEST':
            expect(set(event) == {'ordinal', 'op', 'doc_id', 'path_hex', 'sha256', 'bytes'},
                   'invalid INGEST schema')
            expect(did == len(docs) and did < 2**64 - 1, 'non-monotonic id')
            path = bytes.fromhex(event['path_hex'])
            expect(path_hex(path) == event['path_hex'] and event['path_hex'] not in paths,
                   'duplicate/noncanonical path')
            expect(type(event['bytes']) is int and 0 <= event['bytes'] <= 4096 and
                   type(event['sha256']) is str and len(event['sha256']) == 64 and
                   all(x in '0123456789abcdef' for x in event['sha256']),
                   'invalid source identity')
            docs[did] = {'doc_id': did, 'revision': 0, 'path_hex': event['path_hex'],
                         'sha256': event['sha256'], 'bytes': event['bytes'],
                         'visible': True}
            paths[event['path_hex']] = did
        else:
            expect(did in docs and docs[did]['visible'], 'missing/tombstoned document')
            doc = docs[did]
            if op == 'RENAME':
                expect(set(event) == {'ordinal', 'op', 'doc_id', 'path_hex'},
                       'invalid RENAME schema')
                path = bytes.fromhex(event['path_hex'])
                expect(path_hex(path) == event['path_hex'] and
                       event['path_hex'] not in paths and event['path_hex'] != doc['path_hex'],
                       'duplicate/unchanged/unsafe rename')
                del paths[doc['path_hex']]
                paths[event['path_hex']] = did
                doc['path_hex'] = event['path_hex']
            elif op == 'REVISE':
                expect(set(event) == {'ordinal', 'op', 'doc_id', 'revision', 'sha256', 'bytes'},
                       'invalid REVISE schema')
                expect(type(event['revision']) is int and event['revision'] == doc['revision'] + 1
                       and event['revision'] < 2**64, 'nonmonotonic revision')
                expect(type(event['bytes']) is int and 0 <= event['bytes'] <= 4096 and
                       type(event['sha256']) is str and len(event['sha256']) == 64 and
                       all(x in '0123456789abcdef' for x in event['sha256']) and
                       event['sha256'] != doc['sha256'], 'invalid/unchanged revision')
                doc['revision'] = event['revision']
                doc['sha256'] = event['sha256']
                doc['bytes'] = event['bytes']
            else:
                expect(set(event) == {'ordinal', 'op', 'doc_id'},
                       'invalid TOMBSTONE schema')
                doc['visible'] = False
                del paths[doc['path_hex']]
    return [docs[i].copy() for i in sorted(docs)]


def make_root(events, generation, parent):
    expect(type(generation) is int and generation >= 0, 'bad generation')
    visible = derive(events)
    base = {'schema': PROFILE, 'vault_id': VAULT_ID, 'generation': generation,
            'previous_root_sha256': parent, 'events': events,
            'next_doc_id': len(visible), 'documents': visible,
            'coverage': [[d['doc_id'], d['revision']] for d in visible if d['visible']]}
    base['snapshot_id'] = sha(DOMAIN + canonical(base))
    return base


def object_store(bundle, content):
    expect(type(content) is bytes and len(content) <= 4096, 'fixture byte cap')
    o = bundle / 'objects' / (sha(content) + '.bin')
    if o.exists():
        expect(not o.is_symlink() and o.read_bytes() == content,
               'existing object corrupted')
    else:
        with o.open('xb') as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
    return {'sha256': sha(content), 'bytes': len(content)}


def append(events, operation, doc_id, path=None, content=None):
    entry = {'ordinal': len(events), 'op': operation, 'doc_id': doc_id}
    if operation == 'INGEST':
        entry.update(path_hex=path_hex(path), sha256=sha(content), bytes=len(content))
    elif operation == 'RENAME':
        entry['path_hex'] = path_hex(path)
    elif operation == 'REVISE':
        projection = derive(events)
        expect(doc_id < len(projection), 'wrong doc for revision')
        entry.update(revision=projection[doc_id]['revision'] + 1,
                     sha256=sha(content), bytes=len(content))
    events.append(entry)
    try:
        derive(events)
    except Exception:
        events.pop()
        raise
    return events


def publish(bundle, events, generation, expected_pin):
    """Single writer; no-overwrite roots; CURRENT CAS after fsynced root.

    Crash durability is tested for ordinary exits, not guaranteed on powerloss
    with every possible filesystem. No signed/currentness witness.
    """
    lockfile = bundle / 'LOCK'
    with lockfile.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('concurrent writer denied') from exc
        current_path = bundle / 'CURRENT'
        expect(not current_path.is_symlink(), 'CURRENT symlink rejected')
        present = current_path.read_text().strip() if current_path.exists() else None
        expect(present == expected_pin, 'compare-and-swap stale root')
        root = make_root(events, generation, expected_pin)
        data = canonical(root)
        pin = sha(data)
        record = bundle / 'snapshots' / (pin + '.json')
        expect(not record.exists(), 'duplicate snapshot publication')
        tmp = bundle / 'snapshots' / (pin + '.incomplete')
        with tmp.open('xb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, record)
        tmp.unlink()
        snapshot_fd = os.open(bundle / 'snapshots', os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(snapshot_fd)
        finally:
            os.close(snapshot_fd)
        temp_pointer = bundle / 'CURRENT.incomplete'
        with temp_pointer.open('xb') as out:
            out.write((pin + '\n').encode())
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_pointer, current_path)
        directory_fd = os.open(bundle, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return root, pin


def query(bundle, root, pin, pattern, cap=None):
    expect(type(pattern) is bytes and pattern, 'empty query rejected')
    expect(cap is None or (type(cap) is int and 0 <= cap <= 256), 'bad cap')
    expect(sha(canonical(root)) == pin, 'root identity mismatch')
    visible = [d for d in root['documents'] if d['visible']]
    coordinates = []
    for doc in visible:
        file = bundle / 'objects' / (doc['sha256'] + '.bin')
        expect(file.is_file() and not file.is_symlink() and
               file.stat().st_size == doc['bytes'], 'mandatory source missing')
        raw = file.read_bytes()
        expect(sha(raw) == doc['sha256'], 'source object SHA mismatch')
        start = raw.find(pattern)
        while start != -1:
            coordinates.append([doc['doc_id'], doc['revision'], start])
            start = raw.find(pattern, start + 1)
    count = len(coordinates)
    emitted = coordinates if cap is None else coordinates[:cap]
    limited = len(emitted) != count
    return {'schema': 'GLYPH_APPEND_QUERY_V2_RESEARCH', 'root_sha256': pin,
            'vault_id': root['vault_id'], 'generation': root['generation'],
            'pattern_hex': pattern.hex(), 'count': count,
            'max_offsets': cap, 'coordinates': emitted,
            'coverage': [[d['doc_id'], d['revision']] for d in visible],
            'status': 'UNTRUSTED' if limited else 'FOUND' if count else 'PROVEN_EMPTY',
            'reason': 'LIMITED_COMPLETENESS' if limited else 'FULL_SNAPSHOT_ORACLE_SCAN'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.output
    expect(not bundle.exists(), 'refuse existing bundle')
    bundle.mkdir(parents=True)
    (bundle / 'objects').mkdir()
    (bundle / 'snapshots').mkdir()
    events = []
    pins = []
    states = []
    def snapshot():
        generation = len(pins)
        root, pin = publish(bundle, list(events), generation, pins[-1] if pins else None)
        states.append(root)
        pins.append(pin)
    object_store(bundle, b'banana\x00')
    append(events, 'INGEST', 0, path=b'z.bin', content=b'banana\x00')
    snapshot()  # V0
    object_store(bundle, b'banana\xff')
    append(events, 'INGEST', 1, path=b'a.bin', content=b'banana\xff')
    snapshot()  # V1 late-early sorting path
    append(events, 'RENAME', 0, path=b'q.bin')
    snapshot()  # V2 rename: data unchanged
    object_store(bundle, b'new\xffbanana')
    append(events, 'REVISE', 0, content=b'new\xffbanana')
    snapshot()  # V3 new data revision
    append(events, 'INGEST', 2, path=b'copy.bin', content=b'banana\xff')
    snapshot()  # V4 same bytes, different ID, dedup
    object_store(bundle, b'')
    append(events, 'INGEST', 3, path=b'empty.bin', content=b'')
    snapshot()  # V5 empty source
    append(events, 'TOMBSTONE', 1)
    snapshot()  # V6 tombstone hides but doesn't erase history
    (bundle/'pins.json').write_bytes(canonical(pins))
    print(json.dumps({'status': 'CORE_L10_V2_SOURCE_FIXTURE_GENERATED',
                      'roots': len(pins), 'unique_objects': len(list((bundle/'objects').iterdir())),
                      'final_root_sha256': pins[-1]}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'UNTRUSTED', 'error': str(exc)}), file=sys.stderr)
        sys.exit(3)
