"""Bounded offline copy, not pack-only catalogue reconstruction or a live backup.

Source must be closed. Its owner lock remains held through copy verification.
Keep the returned receipt hash independently. Failed destinations are retained
without a completion receipt; no source files are removed or changed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from packed_checkpoint import _hash_regular

RECEIPT = 'BACKUP.json'


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _files(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink():
                raise ValueError('symlinks not supported in offline backup')
        for name in files:
            p = Path(directory) / name
            if not p.is_file():
                raise ValueError('non-regular backup entry')
            yield p


def create_backup(metadata, data, destination, *, max_bytes=256 * 1024**2, max_files=10000):
    metadata, data, destination = map(lambda p: Path(p).resolve(), (metadata, data, destination))
    paths = (metadata, data, destination)
    if any(a == b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('backup and source directories must be disjoint')
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError('destination must be absent with existing parent')
    if not 0 < max_bytes <= 1024**3 or not 0 < max_files <= 10000:
        raise ValueError('bounded pilot: at most 1 GiB and 10000 files')
    fd = os.open(metadata / 'owner.lock', os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError('invalid owner lock')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allowed = {'owner.lock','index.sqlite'}
        if {p.name for p in metadata.iterdir()} != allowed:
            raise ValueError('metadata must be clean and contain only index.sqlite and owner.lock')
        if not (data/'identity.json').is_file() or not (data/'packs').is_dir():
            raise ValueError('data identity or packs absent')
        sources = []
        size = 0
        for label, root in (('metadata',metadata),('data',data)):
            for p in _files(root):
                size += p.stat().st_size
                sources.append((p, Path(label)/p.relative_to(root)))
                if size > max_bytes or len(sources) > max_files:
                    raise ValueError('backup budget exceeded')
        destination.mkdir(mode=0o700)
        _sync_dir(destination.parent)
        records = []
        total_copied = 0
        for source, relative in sorted(sources):
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            digest, copied = hashlib.sha256(), 0
            with os.fdopen(source_fd,'rb') as incoming, target.open('xb') as outgoing:
                if not stat.S_ISREG(os.fstat(incoming.fileno()).st_mode):
                    raise ValueError('non-regular source')
                for block in iter(lambda: incoming.read(1024**2), b''):
                    copied += len(block)
                    total_copied += len(block)
                    if total_copied > max_bytes:
                        raise ValueError('source grew beyond budget')
                    outgoing.write(block); digest.update(block)
                outgoing.flush(); os.fsync(outgoing.fileno())
            sha = digest.hexdigest()
            if _hash_regular(target) != sha:
                raise ValueError('backup readback mismatch')
            records.append(dict(path=relative.as_posix(), bytes=copied, sha256=sha))
        # Empty packs directories also survive an empty store copy.
        (destination/'data'/'packs').mkdir(parents=True, exist_ok=True)
        for directory, _, _ in os.walk(destination, topdown=False):
            _sync_dir(directory)
        raw = json.dumps({'format':'GLYPH_OFFLINE_BACKUP_V1', 'complete':True,
                          'copy_readback_verified':True, 'store_semantics_verified':False,
                          'files':records},sort_keys=True,separators=(',',':')).encode()
        with (destination/(RECEIPT+'.pending')).open('xb') as output:
            output.write(raw); output.flush(); os.fsync(output.fileno())
        os.rename(destination/(RECEIPT+'.pending'),destination/RECEIPT)
        _sync_dir(destination)
        return hashlib.sha256(raw).hexdigest()


def verify_backup(destination, expected_receipt_sha256):
    """Verify trusted receipt and every copied byte before opening a backup.

    Trusted local paths/process assumed. Not safe against concurrent malicious
    filesystem replacement. Does not assert that the original store was healthy.
    """
    destination = Path(destination)
    receipt = destination/RECEIPT
    if receipt.stat().st_size > 4 * 1024**2:
        raise ValueError('receipt budget exceeded')
    if _hash_regular(receipt) != expected_receipt_sha256:
        raise ValueError('backup receipt mismatch')
    raw = receipt.read_bytes()
    if len(raw) > 4 * 1024**2:
        raise ValueError('receipt budget exceeded')
    doc = json.loads(raw)
    if doc.get('format') != 'GLYPH_OFFLINE_BACKUP_V1' or doc.get('complete') is not True:
        raise ValueError('incomplete backup')
    expected = {RECEIPT}
    for row in doc['files']:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('metadata','data'):
            raise ValueError('invalid backup path')
        if row['path'] in expected:
            raise ValueError('duplicate backup path')
        expected.add(row['path'])
    actual = {p.relative_to(destination).as_posix() for p in _files(destination)}
    if actual != expected:
        raise ValueError('missing or unexpected backup file')
    for row in doc['files']:
        p = destination/row['path']
        if p.stat().st_size != row['bytes'] or _hash_regular(p) != row['sha256']:
            raise ValueError('backup data mismatch')
    return {'copy_readback_verified':True, 'files':len(doc['files']),
            'bytes':sum(row['bytes'] for row in doc['files']), 'store_semantics_verified':False}
