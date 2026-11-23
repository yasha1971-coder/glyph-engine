#!/usr/bin/env python3
"""Standalone CORE-L10 V2 metadata+source-oracle verifier.

NO producer imports. External pins are caller-supplied, unsigned; source blobs
are only fixture oracles, NOT a compressed GLYPH archive or physical index proof.
"""
import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path

PROFILE = 'GLYPH_APPEND_IDENTITY_V2_RESEARCH'
DOMAIN = b'GLYPH_APPEND_SNAPSHOT_V2_RESEARCH\x00'
HEX = '0123456789abcdef'
MAX_DOCS = 48


def fail_if(condition, reason):
    if condition:
        raise ValueError(reason)


def hashbytes(blob):
    return hashlib.sha256(blob).hexdigest()


def serialize(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def paircheck(pairs):
    d = {}
    for k, v in pairs:
        fail_if(k in d, 'duplicate JSON field')
        d[k] = v
    return d


def parse(data):
    x = json.loads(data.decode('utf-8'), object_pairs_hook=paircheck,
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    fail_if(type(x) not in (dict, list) or data != serialize(x), 'not canonical JSON')
    return x


def read_regular(path, limit, allow_empty=False):
    statx = path.lstat()
    fail_if(not stat.S_ISREG(statx.st_mode) or not ((0 if allow_empty else 1) <= statx.st_size <= limit),
            'nonregular/invalid file bounds')
    raw = path.read_bytes()
    fail_if(len(raw) != statx.st_size, 'file changed during read')
    return raw


def path_valid(s):
    if type(s) is not str or len(s)%2 or len(s)==0 or any(x not in HEX for x in s):
        return False
    b = bytes.fromhex(s)
    return 0 < len(b) <= 4096 and not b.startswith(b'/') and b'\x00' not in b \
        and b'\\' not in b and all(c not in (b'', b'.', b'..') for c in b.split(b'/'))


def valid_sha(s):
    return type(s) is str and len(s) == 64 and all(c in HEX for c in s)


def verify_events(events):
    fail_if(type(events) is not list or len(events) > 160, 'events unbounded')
    state, names = {}, {}
    for ordinal, e in enumerate(events):
        fail_if(type(e) is not dict or type(e.get('ordinal')) is not int or
                e['ordinal'] != ordinal or type(e.get('doc_id')) is not int or
                e.get('op') not in ('INGEST', 'RENAME', 'REVISE', 'TOMBSTONE'),
                'event type/order/id invalid')
        did, op = e['doc_id'], e['op']
        if op == 'INGEST':
            fail_if(set(e) != {'op','ordinal','doc_id','path_hex','sha256','bytes'} or
                    did != len(state) or did >= (1<<64)-1 or
                    not path_valid(e.get('path_hex')) or
                    e['path_hex'] in names or not valid_sha(e.get('sha256')) or
                    type(e.get('bytes')) is not int or e['bytes'] < 0 or e['bytes'] > 4096,
                    'invalid ingest or unstable doc_id')
            names[e['path_hex']] = did
            state[did] = dict(doc_id=did, revision=0, path_hex=e['path_hex'],
                              sha256=e['sha256'], bytes=e['bytes'], visible=True)
            continue
        fail_if(did not in state or not state[did]['visible'], 'event targets absent/tombstoned')
        item = state[did]
        if op == 'RENAME':
            fail_if(set(e) != {'op','ordinal','doc_id','path_hex'} or
                    not path_valid(e.get('path_hex')) or
                    e['path_hex'] in names or item['path_hex'] == e['path_hex'],
                    'invalid/duplicate rename')
            del names[item['path_hex']]
            names[e['path_hex']] = did
            item['path_hex'] = e['path_hex']
        elif op == 'REVISE':
            fail_if(set(e) != {'op','ordinal','doc_id','revision','sha256','bytes'} or
                    type(e.get('revision')) is not int or e['revision'] != item['revision'] + 1 or
                    e['revision'] >= (1<<64) or not valid_sha(e.get('sha256')) or
                    e['sha256'] == item['sha256'] or type(e.get('bytes')) is not int or
                    not (0 <= e['bytes'] <= 4096), 'revision mismatch')
            item['revision'] = e['revision']
            item['sha256'] = e['sha256']
            item['bytes'] = e['bytes']
        else:
            fail_if(set(e) != {'op','ordinal','doc_id'}, 'tombstone extra fields')
            del names[item['path_hex']]
            item['visible'] = False
    fail_if(len(state) > MAX_DOCS, 'too many docs for research')
    return [state[i] for i in range(len(state))]


def check_root(raw, expected_pin, generation, older):
    fail_if(not valid_sha(expected_pin) or hashbytes(raw) != expected_pin,
            'root not anchored to expected external pin')
    obj = parse(raw)
    fail_if(type(obj) is not dict or set(obj) != {
        'schema','vault_id','generation','previous_root_sha256','events',
        'next_doc_id','documents','coverage','snapshot_id'}, 'invalid root schema')
    fail_if(obj['schema'] != PROFILE or type(obj['vault_id']) is not str or
            len(obj['vault_id']) != 32 or any(x not in HEX for x in obj['vault_id']) or
            type(obj['generation']) is not int or obj['generation'] != generation,
            'root version/vault mismatch')
    fail_if(obj['previous_root_sha256'] != older, 'root old-pin mismatch')
    expected_projection = verify_events(obj['events'])
    fail_if(obj['documents'] != expected_projection or
            type(obj['next_doc_id']) is not int or
            obj['next_doc_id'] != len(expected_projection) or
            obj['coverage'] != [[d['doc_id'],d['revision']]
                                for d in expected_projection if d['visible']],
            'falsified root projection or incomplete coverage')
    s = dict(obj)
    claimed = s.pop('snapshot_id')
    fail_if(not valid_sha(claimed) or claimed != hashbytes(DOMAIN + serialize(s)),
            'domain-separated snapshot identity invalid')
    return obj


def check_objects(bundle, root):
    objects = {}
    for d in root['documents']:
        if d['visible']:
            path = bundle / 'objects' / (d['sha256'] + '.bin')
            data = read_regular(path, 4097, allow_empty=True)
            fail_if(len(data) != d['bytes'] or hashbytes(data) != d['sha256'],
                    'required source-oracle object missing/corrupt')
            objects[d['doc_id']] = data
    return objects


def verify_all(bundle, expected_pins):
    fail_if(type(expected_pins) is not list or not (1<=len(expected_pins)<=12) or
            any(not valid_sha(x) for x in expected_pins), 'invalid external pin vector')
    found = bundle/'CURRENT'
    fail_if(read_regular(found, 100).decode('ascii').strip() != expected_pins[-1],
            'CURRENT pointer does not match externally expected last root')
    roots=[]
    old_pin=None
    old_events=[]
    vault=None
    for i, pin in enumerate(expected_pins):
        file = bundle/'snapshots'/(pin+'.json')
        root = check_root(read_regular(file, 65536), pin, i, old_pin)
        fail_if(vault is not None and root['vault_id'] != vault,'different vault ID in lineage')
        fail_if(root['events'][:len(old_events)] != old_events or
                (i>0 and len(root['events']) <= len(old_events)),
                'history was changed or not extended')
        check_objects(bundle,root)
        old_pin=pin
        old_events=root['events']
        vault=root['vault_id']
        roots.append(root)
    return roots


def independently_query(bundle, root, pin, claim):
    fail_if(type(claim) is not dict or set(claim) != {
        'schema','root_sha256','vault_id','generation','pattern_hex','count',
        'max_offsets','coordinates','coverage','status','reason'},
        'unknown query receipt fields')
    fail_if(claim['schema'] != 'GLYPH_APPEND_QUERY_V2_RESEARCH' or
            claim['root_sha256'] != pin or claim['vault_id'] != root['vault_id'] or
            claim['generation'] != root['generation'], 'query bound to wrong root/version')
    pattern_hex = claim['pattern_hex']
    fail_if(type(pattern_hex) is not str or len(pattern_hex) == 0 or
            len(pattern_hex)%2 or any(c not in HEX for c in pattern_hex),
            'invalid query bytes')
    pattern = bytes.fromhex(pattern_hex)
    cap = claim['max_offsets']
    fail_if(cap is not None and (type(cap) is not int or not (0<=cap<=256)),
            'invalid complete/limited policy')
    docbytes = check_objects(bundle,root)
    all_coords = []
    for did in sorted(docbytes):
        raw=docbytes[did]
        rev=root['documents'][did]['revision']
        i=raw.find(pattern)
        while i!=-1:
            all_coords.append([did,rev,i])
            i=raw.find(pattern,i+1)
    coords = all_coords if cap is None else all_coords[:cap]
    limited = len(coords)<len(all_coords)
    status = 'UNTRUSTED' if limited else 'FOUND' if all_coords else 'PROVEN_EMPTY'
    reason = 'LIMITED_COMPLETENESS' if limited else 'FULL_SNAPSHOT_ORACLE_SCAN'
    expected_coverage = [[d['doc_id'],d['revision']] for d in root['documents'] if d['visible']]
    fail_if(claim['coverage'] != expected_coverage or
            type(claim['count']) is not int or claim['count'] != len(all_coords) or
            claim['coordinates'] != coords or claim['status'] != status or
            claim['reason'] != reason,
            'false match/empty/coverage or limited completeness')
    return {'status':'CORE_L10_V2_INDEPENDENT_REPLAY_PASS',
            'outcome':status,'count':len(all_coords),
            'root_sha256':pin,'source_oracle_bytes':sum(len(x) for x in docbytes.values())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--expected-pins-json',required=True)
    p.add_argument('--generation',type=int,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    try:
        pins=json.loads(a.expected_pins_json)
        roots=verify_all(a.bundle,pins)
        fail_if(a.generation<0 or a.generation>=len(roots), 'no such generation')
        claim,_=None,None
        claim=parse(read_regular(a.receipt,65536))
        print(json.dumps(independently_query(a.bundle,roots[a.generation],
                                             pins[a.generation],claim),sort_keys=True))
        return 0
    except Exception as e:
        print(json.dumps({'status':'UNTRUSTED','reason':str(e)[:260]},sort_keys=True),file=sys.stderr)
        return 3


if __name__=='__main__':
    sys.exit(main())
