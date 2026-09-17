"""Bounded recovery recipes independent of SQLite. Synthetic/offline pilot.

Keep capsule bytes and their trusted SHA-256 independently. Original data packs
remain necessary. No mutation, reconstruction guesses, or filesystem output.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct

from chunk_versions import unpack

FORMAT = 'GLYPH_RECOVERY_RECIPES_V1'
HEADER = struct.Struct('<8sII32s')
MAX_CAPSULE = 8 * 1024**2
MAX_VERSION = 64 * 1024**2


def _regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink in recovery path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('not a regular recovery file')
    return os.fdopen(fd, 'rb')


def load(raw, expected):
    if len(raw) > MAX_CAPSULE or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('recovery capsule pin or size mismatch')
    doc = json.loads(raw)
    if doc.get('format') != FORMAT or not isinstance(doc.get('versions'), list):
        raise ValueError('unknown recovery format')
    ids = [v['id'] for v in doc['versions']]
    if len(ids) > 1000 or len(set(ids)) != len(ids):
        raise ValueError('invalid version catalogue')
    return doc


def restore(raw, expected, data_root, version_id):
    """Return a complete verified version in memory; never a partial file."""
    doc = load(raw, expected)
    with _regular(Path(data_root)/'identity.json') as f:
        identity = json.loads(f.read(4097))
    if identity != doc['identity']:
        raise ValueError('wrong data store')
    matches = [v for v in doc['versions'] if v['id'] == version_id]
    if len(matches) != 1:
        raise ValueError('version absent')
    v = matches[0]
    if not isinstance(v['bytes'], int) or not 0 <= v['bytes'] <= MAX_VERSION or len(v['blocks']) > 10000:
        raise ValueError('version recovery budget')
    output = bytearray()
    for b in v['blocks']:
        tx, number, offset, size, stored = (b[k] for k in ('tx','pack','offset','bytes','stored'))
        if not re.fullmatch('[0-9a-f]{32}',tx) or not all(type(x) is int for x in (number,offset,size,stored)):
            raise ValueError('bad pack locator')
        if not (0 <= number < 100000000 and offset >= 0 and 0 < size <= 4*1024**2 and 0 < stored <= size+1):
            raise ValueError('pack bounds')
        if len(output)+size > v['bytes']:
            raise ValueError('version size exceeded')
        digest = bytes.fromhex(b['sha256'])
        if len(digest) != 32:
            raise ValueError('bad block hash')
        path = Path(data_root)/'packs'/tx/f'{number:08d}.pack'
        with _regular(path) as f:
            f.seek(offset)
            header, payload = f.read(HEADER.size), f.read(stored)
        if header != HEADER.pack(b'GLYPHPK1',size,stored,digest) or len(payload) != stored:
            raise ValueError('pack framing mismatch')
        block = unpack(payload,size)
        if len(block) != size or hashlib.sha256(block).digest() != digest:
            raise ValueError('block checksum mismatch')
        output.extend(block)
    if len(output) != v['bytes'] or hashlib.sha256(output).hexdigest() != v['sha256']:
        raise ValueError('version checksum mismatch')
    return bytes(output)


def export(store):
    """Caller holds Store's lifetime owner lock; no transaction may be pending.

    Export all versions in this small store or fail, never silently omit versions.
    Current schema only. Does not change the existing writer's commit protocol.
    """
    if store.db.in_transaction or (store.meta/'pending.json').exists():
        raise ValueError('finish transaction before export')
    rows = store.db.execute('SELECT id,name,created_ns,n,sha,profile FROM versions ORDER BY id LIMIT 1001').fetchall()
    if len(rows) > 1000 or sum(r[3] for r in rows) > 128*1024**2:
        raise ValueError('capsule export workload budget')
    versions = []
    for ident,name,created,size,digest,profile in rows:
        refs = store.db.execute('SELECT seq,sha FROM refs WHERE version=? ORDER BY seq LIMIT 10001',(ident,)).fetchall()
        if len(refs)>10000 or [r[0] for r in refs] != list(range(len(refs))):
            raise ValueError('invalid or excessive version recipe')
        blocks=[]
        for _,sha in refs:
            obj=store.db.execute('SELECT tx,pack,off,n,stored FROM objects WHERE sha=?',(sha,)).fetchone()
            if obj is None:
                raise ValueError('missing object locator')
            blocks.append(dict(zip(('tx','pack','offset','bytes','stored'),obj),sha256=sha.hex()))
        versions.append(dict(id=ident,name=name,created_ns=created,bytes=size,sha256=digest.hex(),profile=profile,blocks=blocks))
    raw=json.dumps(dict(format=FORMAT,identity=store.identity,versions=versions),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    pin=hashlib.sha256(raw).hexdigest()
    load(raw,pin)
    # Independent reader proves every exported recipe reconstructs the stored hash.
    for v in versions:
        restore(raw,pin,store.data,v['id'])
    return raw,pin
