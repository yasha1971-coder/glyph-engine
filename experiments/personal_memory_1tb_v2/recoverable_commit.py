"""Opt-in bounded commit wrapper; existing Store writer is unchanged.

Caller owns the Store lifetime lock. Recovery root is private to this store.
An exception after DB commit is ambiguous: reconcile with publish(), do not
blindly repeat the write. Never claims transactional rollback across two stores.
"""
import hashlib
import os
from pathlib import Path
import tempfile

import recovery_recipes as recipes
from packed_backup import _sync_dir


def _root(store, root):
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('symlink recovery root')
    for source in (store.meta.resolve(), store.data.resolve()):
        if root == source or root in source.parents or source in root.parents:
            raise ValueError('recovery root must be separate')
    if not root.exists():
        root.mkdir(mode=0o700)
        _sync_dir(root.parent)
    if (root/'HEAD').exists():
        raw, _ = current(root)
        import json
        if json.loads(raw)['identity'] != store.identity:
            raise ValueError('recovery root belongs to another store')
    elif any(root.iterdir()):
        raise ValueError('uninitialized nonempty recovery root; inspect before reuse')
    return root


def current(root, *, expected_pin=None):
    """Read only published HEAD. External expected pin prevents silent rollback."""
    root = Path(root)
    with recipes._regular(root/'HEAD') as f:
        pin = f.read(66).decode().strip()
    import re
    if not re.fullmatch('[0-9a-f]{64}',pin) or (expected_pin is not None and expected_pin != pin):
        raise ValueError('invalid or unexpected recovery head')
    with recipes._regular(root/(pin+'.json')) as f:
        raw = f.read(recipes.MAX_CAPSULE+1)
    recipes.load(raw,pin)
    return raw,pin


def publish(store, root, *, fault=lambda stage: None):
    """Reconcile the whole bounded committed catalogue; caller holds Store lock."""
    root = _root(store,root)
    raw,pin = recipes.export(store)
    target = root/(pin+'.json')
    fd,name = tempfile.mkstemp(prefix='.capsule-',dir=root)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        fault('after_capsule_fsync')
        if target.exists():
            with recipes._regular(target) as f:
                if hashlib.sha256(f.read(recipes.MAX_CAPSULE+1)).hexdigest()!=pin:
                    raise ValueError('existing capsule corrupt')
        else:
            os.link(name,target)
        _sync_dir(root)
        fault('after_capsule_publish')
        hfd,hname = tempfile.mkstemp(prefix='.head-',dir=root)
        try:
            with os.fdopen(hfd,'w') as f:
                f.write(pin+'\n');f.flush();os.fsync(f.fileno())
            os.replace(hname,root/'HEAD')
            _sync_dir(root)
            fault('after_head_publish')
        finally:
            if os.path.exists(hname):os.unlink(hname)
        return pin
    finally:
        if os.path.exists(name):os.unlink(name)


def commit_version(store, root, name, chunks, *, fault=lambda stage: None, **options):
    # Establish a baseline before accepting the first new version.
    root = _root(store,root)
    if not (root/'HEAD').exists():publish(store,root)
    version = store.put_version(name,chunks,**options)
    fault('after_store_commit')
    pin = publish(store,root,fault=fault)
    return {'version':version,'recovery_pin':pin,'recovery_published':True}
