"""Offline data backup for the actual incremental-memory application.

Trusted local filesystem, cooperating writers. No encryption or code backup.
Never execute a decoder supplied by the backup; caller supplies a trusted one.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

import incremental_memory as inc
import permanent_delete
from packed_backup import _files, _sync_dir
from packed_checkpoint import _hash_regular

RECEIPT = 'PERSONAL_BACKUP.json'
FORMAT = 'GLYPH_PERSONAL_BACKUP_V1'


def disjoint(a, b):
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('source and destination must be disjoint')


def inventory(root, max_bytes):
    files, size = [], 0
    for p in _files(root):
        size += p.stat().st_size
        files.append(p)
        if size > max_bytes or len(files) > 100000:
            raise ValueError('backup inventory budget exceeded')
    return sorted(files)


def copy_file(source, target, budget):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    sha, size = hashlib.sha256(), 0
    with os.fdopen(fd, 'rb') as src, target.open('xb') as dst:
        before = os.fstat(src.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('regular source required')
        for block in iter(lambda: src.read(1024**2), b''):
            size += len(block)
            if size > budget:
                raise ValueError('copy budget exceeded')
            sha.update(block)
            dst.write(block)
        after = os.fstat(src.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('source changed during copy')
        dst.flush()
        os.fsync(dst.fileno())
    if _hash_regular(target) != sha.hexdigest():
        raise ValueError('copy readback mismatch')
    return {'bytes': size, 'sha256': sha.hexdigest()}


def verify_history(memory, max_bytes):
    head = permanent_delete.head(memory)
    seen, total, count = set(), 0, 0
    for pin, doc in permanent_delete.timeline(memory, head):
        for item in doc['files'].values():
            key = (item['storage'], item['sha256'], item['bytes'], item.get('recipe'))
            if key in seen:
                continue
            seen.add(key)
            total += item['bytes']
            if total > max_bytes or len(seen) > 100000:
                raise ValueError('decoded verification budget exceeded')
            data = memory.read_item(item)
            if len(data) != item['bytes'] or inc.digest(data) != item['sha256']:
                raise ValueError('restored history mismatch')
            count += 1
    return {'head': head, 'unique_items_verified': count, 'decoded_bytes': total}


def create(memory, destination, max_bytes=1024**3):
    if not 0 < max_bytes <= 64 * 1024**3:
        raise ValueError('supported budget: 1 byte through 64 GiB')
    destination = Path(destination).absolute()
    if os.path.lexists(destination) or not destination.parent.is_dir():
        raise ValueError('destination must be absent with existing parent')
    destination = destination.resolve()
    for root in (memory.root, memory.backend.root):
        disjoint(root, destination)
    with ExitStack() as stack:
        for name in ('browser.lock', 'writer.lock'):
            fd = os.open(memory.root / name, os.O_RDONLY | os.O_NOFOLLOW)
            lock = stack.enter_context(os.fdopen(fd, 'rb'))
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError('invalid lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.path.lexists(memory.root / permanent_delete.JOURNAL):
            raise ValueError('pending deletion: recover in application first')
        head = permanent_delete.head(memory)
        memory.snapshot(head)
        sources = []
        for label, root in (('memory', memory.root), ('archive', memory.backend.root)):
            for p in inventory(root, max_bytes):
                sources.append((p, Path(label)/p.relative_to(root)))
        if len(sources) > 100000 or sum(p.stat().st_size for p, _ in sources) > max_bytes:
            raise ValueError('total copy budget exceeded')
        destination.mkdir(mode=0o700)
        _sync_dir(destination.parent)
        records, remaining = [], max_bytes
        for source, relative in sources:
            record = copy_file(source, destination/relative, remaining)
            remaining -= record['bytes']
            records.append(dict(path=relative.as_posix(), **record))
        for name in ('objects', 'snapshots', 'chunks', 'recipes'):
            if (memory.root/name).is_dir():
                (destination/'memory'/name).mkdir(exist_ok=True)
        restored = inc.Memory(destination/'memory', destination/'archive', memory.base_pin,
                              memory.backend.precompressor, memory.backend.binary_sha256)
        verified = verify_history(restored, max_bytes)
        if verified['head'] != head:
            raise ValueError('head changed')
        doc = {'format': FORMAT, 'complete': True, 'files': records,
               'archive_sha256': memory.base_pin,
               'precompressor_sha256': memory.backend.binary_sha256,
               'history_verified': verified, 'encrypted': False,
               'code_included': False, 'decoder_included': False}
        for directory, _, _ in os.walk(destination, topdown=False):
            _sync_dir(directory)
        raw = json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()
        if len(raw) > 32 * 1024**2:
            raise ValueError('receipt budget exceeded')
        inc.publish(destination/RECEIPT, raw)
        return hashlib.sha256(raw).hexdigest()


def verify(destination, trusted_pin):
    root = Path(destination)
    raw = inc.read_regular(root, RECEIPT, 32 * 1024**2)
    if inc.digest(raw) != trusted_pin:
        raise ValueError('backup receipt differs from independently retained pin')
    doc = json.loads(raw)
    if doc['format'] != FORMAT or doc['complete'] is not True or len(doc['files']) > 100000:
        raise ValueError('invalid backup receipt')
    expected = {RECEIPT}
    for row in doc['files']:
        p = Path(row['path'])
        if (not p.parts or p.is_absolute() or '..' in p.parts or
                p.parts[0] not in ('memory', 'archive') or p.as_posix() != row['path'] or
                row['path'] in expected):
            raise ValueError('invalid backup path')
        expected.add(row['path'])
    if {p.relative_to(root).as_posix() for p in _files(root)} != expected:
        raise ValueError('missing or unexpected backup files')
    for row in doc['files']:
        p = root/row['path']
        if p.stat().st_size != row['bytes'] or _hash_regular(p) != row['sha256']:
            raise ValueError('backup bytes mismatch')
    return doc


def restore(backup, trusted_pin, destination, precompressor=None, max_bytes=1024**3):
    backup, destination = Path(backup).resolve(), Path(destination).absolute()
    disjoint(backup, destination.resolve())
    if os.path.lexists(destination) or not destination.parent.is_dir():
        raise ValueError('restore destination must be new')
    doc = verify(backup, trusted_pin)
    if not 0 < max_bytes <= 64*1024**3 or sum(r['bytes'] for r in doc['files']) > max_bytes:
        raise ValueError('restore budget exceeded')
    destination.mkdir(mode=0o700)
    _sync_dir(destination.parent)
    remaining = max_bytes
    for row in doc['files']:
        got = copy_file(backup/row['path'], destination/row['path'], remaining)
        remaining -= got['bytes']
        if got != {'bytes': row['bytes'], 'sha256': row['sha256']}:
            raise ValueError('backup changed during restore')
    for name in ('objects', 'snapshots', 'chunks', 'recipes'):
        (destination/'memory'/name).mkdir(exist_ok=True)
    m = inc.Memory(destination/'memory', destination/'archive', doc['archive_sha256'],
                   precompressor, doc['precompressor_sha256'])
    verified = verify_history(m, max_bytes)
    if verified != doc['history_verified']:
        raise ValueError('restored history differs')
    for directory, _, _ in os.walk(destination, topdown=False):
        _sync_dir(directory)
    inc.publish(destination/'RESTORE_COMPLETE.json', json.dumps(verified, sort_keys=True).encode())
    return verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['create','verify','restore'])
    p.add_argument('--memory', type=Path)
    p.add_argument('--archive', type=Path)
    p.add_argument('--archive-sha256')
    p.add_argument('--precompressor', type=Path)
    p.add_argument('--precompressor-sha256')
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--backup-sha256')
    p.add_argument('--destination', type=Path)
    p.add_argument('--max-bytes', type=int, default=1024**3)
    a = p.parse_args()
    if a.action == 'create':
        if not all((a.memory,a.archive,a.archive_sha256)):
            p.error('create requires memory, archive and archive-sha256')
        m = inc.Memory(a.memory,a.archive,a.archive_sha256,a.precompressor,a.precompressor_sha256)
        print(json.dumps({'backup_sha256': create(m,a.backup,a.max_bytes)}))
    elif a.action == 'verify':
        doc = verify(a.backup,a.backup_sha256)
        print(json.dumps({'copy_verified':True,'files':len(doc['files'])}))
    else:
        if a.destination is None:
            p.error('restore requires destination')
        print(json.dumps(restore(a.backup,a.backup_sha256,a.destination,a.precompressor,a.max_bytes)))


if __name__ == '__main__':
    main()
