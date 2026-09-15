"""Offline checkpoint guard. Caller keeps the returned pin independently.

Use only after successful store close/scrub, before reopening a frozen backup.
This verifies metadata identity, not payload health or hardware durability.
Never auto-accept a newly observed pin in place of an expected one.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def _hash_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('checkpoint requires a regular file')
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def checkpoint(metadata, data, *, expected=None):
    """Read only; cooperating writer excluded for the entire hash operation.

    A changed/older checkpoint is rejected, including legitimate later writes.
    This is an exact frozen-copy guard, not automatic live-store rollback detection.
    No security against hostile same-user replacement of parent directories.
    """
    if expected is not None and not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('invalid trusted checkpoint pin')
    metadata, data = Path(metadata), Path(data)
    fd = os.open(metadata / 'owner.lock', os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError('invalid lock file')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('pending.json', 'index.sqlite-journal', 'index.sqlite-wal', 'index.sqlite-shm'):
            if os.path.lexists(metadata / name):
                raise ValueError('store needs recovery or clean closure before checkpoint')
        record = {'format': 'GLYPH_OFFLINE_CHECKPOINT_V1',
                  'index_sha256': _hash_regular(metadata / 'index.sqlite'),
                  'data_identity_sha256': _hash_regular(data / 'identity.json')}
        pin = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if expected is not None and pin != expected:
            raise ValueError('checkpoint mismatch: stale, modified or different metadata')
        return dict(record, checkpoint_sha256=pin, payload_verified=False)
