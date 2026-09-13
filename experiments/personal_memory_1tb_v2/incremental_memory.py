#!/usr/bin/env python3
"""Linux private pilot: pinned base archive + immutable, additive snapshots.

Local trusted repository only. No deletion, model, full-text index or encryption.
Base archive must remain available. New files <= 8 MiB, <= 10000 paths/snapshot.
"""
import argparse
import base64
import contextlib
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
import zlib
import time
import chunk_versions

import verified_hybrid_archive as base
from compressed_preservation import CompressedPreservation
from local_memory_bridge import HEX, read_regular

LEGACY_FORMAT = 'GLYPH_INCREMENTAL_MEMORY_PILOT_V1'
FORMAT = 'GLYPH_INCREMENTAL_MEMORY_PILOT_V2'
LIMIT = 8 * 1024 * 1024
META_LIMIT = 16 * 1024 * 1024
Error = base.ArchiveError


def digest(data):
    return base.sha256_bytes(data)


def valid_path(name):
    if not isinstance(name, str) or not name or name.startswith('/') or any(
            part in ('', '.', '..') for part in name.split('/')):
        raise Error('invalid relative file path')
    return name


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, data):
    """Never replace an immutable object; fsync before exposing its name."""
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise Error('existing immutable object differs')
        fsync_dir(path.parent)
    finally:
        os.unlink(temporary)


class Memory:
    def __init__(self, root, archive, archive_sha256, precompressor=None, binary_sha256=None):
        self.root = Path(root).resolve()
        self.backend = CompressedPreservation(archive, archive_sha256, precompressor, binary_sha256)
        if self.root == self.backend.root or self.backend.root in self.root.parents or self.root in self.backend.root.parents:
            raise Error('memory and base archive must be disjoint')
        self.base_pin = archive_sha256
        self.by_digest = {}
        for name, item in self.backend.view.files.items():
            valid_path(name)
            self.by_digest.setdefault(item['sha256'], name)

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=False)
        os.chmod(self.root, 0o700)
        for name in ('objects', 'snapshots'):
            (self.root / name).mkdir()
        (self.root / 'writer.lock').touch(mode=0o600)
        records = {p: {'sha256': f['sha256'], 'bytes': f['bytes'], 'storage': 'base'}
                   for p, f in self.backend.view.files.items()}
        return self.commit(None, records)

    def commit(self, parent, files):
        raw = base.canonical_json({'format': FORMAT, 'base': self.base_pin,
                                   'parent': parent, 'created_ns': time.time_ns(), 'files': files})
        if len(raw) > META_LIMIT or len(files) > 10000:
            raise Error('snapshot metadata budget exceeded')
        packed = base.canonical_json({'format': 'GLYPH_SNAPSHOT_PACK_V1',
             'deflate_base64': base64.b64encode(zlib.compress(raw, 6)).decode('ascii')})
        if len(packed) < len(raw):
            raw = packed
        pin = digest(raw)
        publish(self.root / 'snapshots' / (pin + '.json'), raw)
        return pin

    def snapshot(self, pin):
        if not isinstance(pin, str) or not HEX.fullmatch(pin):
            raise Error('invalid snapshot pin')
        raw = read_regular(self.root, 'snapshots/' + pin + '.json', META_LIMIT)
        if digest(raw) != pin:
            raise Error('snapshot corruption')
        doc = json.loads(raw)
        if doc.get('format') == 'GLYPH_SNAPSHOT_PACK_V1':
            payload = base64.b64decode(doc['deflate_base64'], validate=True)
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(payload, META_LIMIT + 1)
            if len(decoded) > META_LIMIT or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise Error('invalid packed snapshot')
            doc = json.loads(decoded)
        if doc['format'] not in (FORMAT, LEGACY_FORMAT) or doc['base'] != self.base_pin:
            raise Error('wrong snapshot or base')
        if doc['parent'] is not None and not HEX.fullmatch(doc['parent']):
            raise Error('invalid parent')
        if len(doc['files']) > 10000:
            raise Error('too many files')
        for name, item in doc['files'].items():
            valid_path(name)
            if (not HEX.fullmatch(item['sha256']) or type(item['bytes']) is not int
                    or item['bytes'] < 0 or item['storage'] not in ('base', 'raw', 'deflate', 'chunks-v1')):
                raise Error('invalid file identity')
            if item['storage'] != 'base' and item['bytes'] > LIMIT:
                raise Error('file budget exceeded')
        return doc

    def read_item(self, item):
        if item['storage'] == 'chunks-v1':
            return chunk_versions.read(self, item)
        sha, size = item['sha256'], item['bytes']
        if item['storage'] == 'base':
            name = self.by_digest.get(sha)
            if name is None:
                raise Error('base object missing')
            return self.backend.read_verified(name, size, sha)
        payload = read_regular(self.root, 'objects/' + sha, LIMIT)
        if item['storage'] == 'raw':
            data = payload
        else:
            dec = zlib.decompressobj()
            data = dec.decompress(payload, size + 1)
            if not dec.eof or dec.unused_data or dec.unconsumed_tail:
                raise Error('invalid compressed object')
        if len(data) != size or digest(data) != sha:
            raise Error('object corruption')
        return data

    def restore(self, pin, name, output):
        item = self.snapshot(pin)['files'][name]
        data = self.read_item(item)
        # Selected bytes are verified before a destination is created.
        with Path(output).open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return {'bytes': len(data), 'sha256': digest(data)}

    @contextlib.contextmanager
    def writer(self):
        with (self.root / 'writer.lock').open('rb') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Error('another writer is active') from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def add(self, parent, source, source_modified_ms=None):
        if source_modified_ms is not None and (type(source_modified_ms) is not int or not 0 <= source_modified_ms <= 253402300799000):
            raise Error("invalid browser modification date")
        source = Path(source).resolve()
        if not source.is_dir() or any(a == b or a in b.parents or b in a.parents
                for a, b in ((source, self.root), (source, self.backend.root))):
            raise Error('source must be a separate directory')
        with self.writer():
            files = dict(self.snapshot(parent)['files'])
            known = {v['sha256']: v for v in files.values()}
            # Include historical objects so a reverted file reuses old bytes.
            ancestor = self.snapshot(parent)['parent']
            visited = {parent}
            while ancestor is not None:
                if ancestor in visited or len(visited) >= 1000:
                    raise Error('invalid or oversized history')
                visited.add(ancestor)
                doc = self.snapshot(ancestor)
                for item in doc['files'].values():
                    known.setdefault(item['sha256'], item)
                ancestor = doc['parent']
            count = 0
            def scan_error(exc):
                raise exc
            for directory, dirs, names in os.walk(source, followlinks=False, onerror=scan_error):
                for name in dirs + names:
                    if (Path(directory) / name).is_symlink():
                        raise Error('source symlink rejected')
                for name in sorted(names):
                    count += 1
                    if count > 10000:
                        raise Error('input budget exceeded')
                    path = Path(directory) / name
                    rel = valid_path(path.relative_to(source).as_posix())
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, 'rb') as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
                            raise Error('unsupported input or file exceeds 8 MiB')
                        data = stream.read(LIMIT + 1)
                        after = os.fstat(stream.fileno())
                    if (len(data) != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                        raise Error('source changed during reading')
                    sha = digest(data)
                    if sha in known:
                        item = known[sha]
                        if self.read_item(item) != data:
                            raise Error('existing object mismatch')
                    else:
                        item = chunk_versions.store(self, data, files.get(rel))
                        known[sha] = item
                    previous = files.get(rel)
                    if previous and previous['sha256'] == sha:
                        # A repeated byte-identical upload does not rewrite dates.
                        item = previous
                    else:
                        item = {key: value for key, value in item.items()
                                if key in ('sha256', 'bytes', 'storage', 'recipe')}
                        item.update(saved_ns=time.time_ns(), source_modified_ms=source_modified_ms,
                                    source_created_ms=None,
                                    date_source='browser-reported' if source_modified_ms is not None else 'unknown')
                    files[rel] = item
            if files == self.snapshot(parent)['files']:
                return parent
            if len(visited) >= 1000:
                raise Error('history budget exceeded; current snapshot preserved')
            return self.commit(parent, files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'add', 'list', 'restore'])
    p.add_argument('--memory', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--precompressor', type=Path)
    p.add_argument('--precompressor-sha256')
    p.add_argument('--snapshot')
    p.add_argument('--source', type=Path)
    p.add_argument('--path')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    m = Memory(a.memory, a.archive, a.archive_sha256, a.precompressor, a.precompressor_sha256)
    if a.action == 'init':
        result = {'snapshot': m.initialize()}
    elif a.action == 'add':
        if a.source is None:
            p.error('--source required')
        result = {'snapshot': m.add(a.snapshot, a.source)}
    elif a.action == 'list':
        result = m.snapshot(a.snapshot)
    else:
        if a.path is None or a.output is None:
            p.error('--path and --output required')
        result = m.restore(a.snapshot, a.path, a.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
