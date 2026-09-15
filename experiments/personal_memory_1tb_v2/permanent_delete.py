"""Explicit deletion from the active overlay; external base archive is read-only.

Caller owns the browser lock. Journal recovery runs before any UI request.
Not secure erasure of backups, filesystem free space, or shared legacy objects.
"""
import base64
import json
import os
import stat
import tempfile
import zlib
import incremental_memory as inc
import chunk_versions

JOURNAL = 'DELETE_PENDING.json'
BUDGET = 64 * 1024**2


def head(m):
    return inc.read_regular(m.root, 'CURRENT', 65).decode().strip()


def set_head(m, pin):
    fd, name = tempfile.mkstemp(prefix='.head-', dir=m.root)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(pin + '\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, m.root / 'CURRENT')
        inc.fsync_dir(m.root)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def timeline(m, pin):
    result, seen, size = [], set(), 0
    while pin is not None:
        if pin in seen or len(seen) >= 1000:
            raise inc.Error('invalid history')
        seen.add(pin)
        doc = m.snapshot(pin)
        size += len(inc.base.canonical_json(doc))
        if size > BUDGET:
            raise inc.Error('deletion metadata budget exceeded')
        result.append((pin, doc)); pin = doc['parent']
    return list(reversed(result))


def references(m, docs):
    result, seen = set(), set()
    for doc in docs:
        for item in doc['files'].values():
            key = (item['storage'], item['sha256'], item.get('recipe'))
            if key in seen:
                continue
            seen.add(key)
            if item['storage'] in ('raw', 'deflate'):
                result.add('objects/' + item['sha256'])
            elif item['storage'] == 'chunks-v1':
                result.add('recipes/' + item['recipe'])
                for c in chunk_versions.recipe(m, item)['chunks']:
                    if 'source' not in c:
                        result.add('chunks/' + c['sha256'])
                    elif c['source']['storage'] != 'base':
                        result.add('objects/' + c['source']['sha256'])
    return result


def checked_unlink(m, relative):
    pieces = relative.split('/')
    if len(pieces) != 2 or pieces[0] not in ('snapshots', 'objects', 'chunks', 'recipes'):
        raise inc.Error('invalid cleanup target')
    name = pieces[1][:-5] if pieces[0] == 'snapshots' and pieces[1].endswith('.json') else pieces[1]
    if not inc.HEX.fullmatch(name):
        raise inc.Error('invalid cleanup identity')
    directory = m.root / pieces[0]
    if directory.is_symlink() or not directory.is_dir():
        raise inc.Error('invalid storage directory')
    target = directory / pieces[1]
    try:
        info = target.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        raise inc.Error('invalid cleanup file')
    target.unlink()
    inc.fsync_dir(directory)


def _recover(m):
    p = m.root / JOURNAL
    if not p.exists() and not p.is_symlink():
        return
    plan = json.loads(inc.read_regular(m.root, JOURNAL, inc.META_LIMIT))
    if plan.get('format') != 'GLYPH_DELETE_TRANSACTION_V1':
        raise inc.Error('invalid deletion journal')
    current = head(m)
    if current == plan['old']:
        targets = ['snapshots/' + x + '.json' for x in plan['new_snapshots']]
    elif current == plan['new']:
        # Validate the surviving chain and dependency graph before cleanup.
        chain = timeline(m, current)
        keep_pins = {p for p, _ in chain}
        keep_refs = references(m, [d for _, d in chain])
        if keep_pins.intersection(plan['old_snapshots']) or keep_refs.intersection(plan['payloads']):
            raise inc.Error('unsafe deletion plan')
        targets = ['snapshots/' + x + '.json' for x in plan['old_snapshots']] + plan['payloads']
    else:
        raise inc.Error('deletion journal and CURRENT disagree')
    for target in targets:
        checked_unlink(m, target)
    p.unlink(); inc.fsync_dir(m.root)


def recover(m):
    with m.writer():
        _recover(m)


def delete(m, expected, path, selected=None):
    """selected=None removes all history; otherwise one contiguous content version.

    Deleting the current version makes the previous surviving version current.
    Older global snapshot IDs become invalid after this explicit destructive edit.
    """
    inc.valid_path(path)
    with m.writer():
        _recover(m)
        if head(m) != expected:
            raise inc.Error('stale deletion request')
        chain = timeline(m, expected)
        if path not in chain[-1][1]['files']:
            raise inc.Error('unknown document')
        pins = {p for p, _ in chain}
        disk = {p.name for p in (m.root / 'snapshots').iterdir()}
        if disk != {p + '.json' for p in pins}:
            raise inc.Error('independent snapshots require explicit maintenance before deletion')
        start, stop = 0, len(chain)
        if selected is not None:
            positions = [i for i, (p, _) in enumerate(chain) if p == selected]
            if not positions or path not in chain[positions[0]][1]['files']:
                raise inc.Error('unknown version')
            start = positions[0]; stop = start + 1
            def identity(item):
                return item.get('sha256'), item.get('saved_ns')
            key = identity(chain[start][1]['files'][path])
            while start and identity(chain[start-1][1]['files'].get(path, {})) == key:
                start -= 1
            while stop < len(chain) and identity(chain[stop][1]['files'].get(path, {})) == key:
                stop += 1
        old_refs = references(m, [doc for _, doc in chain])
        fallback = chain[start-1][1]['files'].get(path) if selected is not None and start else None
        docs, blobs, parent = [], {}, None
        for i, (_, old) in enumerate(chain):
            doc = dict(old, parent=parent, files=dict(old['files']))
            if start <= i < stop:
                if fallback is None:
                    doc['files'].pop(path, None)
                else:
                    doc['files'][path] = fallback
            raw = inc.base.canonical_json(doc)
            packed = inc.base.canonical_json({'format': 'GLYPH_SNAPSHOT_PACK_V1',
                      'deflate_base64': base64.b64encode(zlib.compress(raw, 6)).decode()})
            raw = min((raw, packed), key=len)
            pin = inc.digest(raw)
            blobs[pin] = raw; docs.append(doc); parent = pin
        keep_refs = references(m, docs)
        # Unchanged prefix snapshots can have identical hashes; preserve them.
        new_only = set(blobs) - pins
        obsolete = pins - set(blobs)
        plan = {'format': 'GLYPH_DELETE_TRANSACTION_V1', 'old': expected, 'new': parent,
                'old_snapshots': sorted(obsolete), 'new_snapshots': sorted(new_only),
                'payloads': sorted(old_refs - keep_refs)}
        raw_plan = inc.base.canonical_json(plan)
        if len(raw_plan) > inc.META_LIMIT:
            raise inc.Error('deletion journal budget exceeded')
        inc.publish(m.root / JOURNAL, raw_plan)
        # Once the journal exists, restart either aborts or completes safely.
        for pin in sorted(new_only):
            inc.publish(m.root / 'snapshots' / (pin + '.json'), blobs[pin])
        set_head(m, parent)
        _recover(m)
        return parent
