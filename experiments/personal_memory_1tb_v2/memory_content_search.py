"""Read-only search adapter for a pinned incremental-memory snapshot."""
from indexed_memory import ContentIndex
import time


def search(memory, pin, query, *, history=False, casefold=False):
    started = time.monotonic()
    class View:
        def __init__(self, version, files):
            self.version, self.files = version, files

        def read(self, path):
            item = self.files[path]
            if item['bytes'] > 8 * 1024**2:
                return None
            return memory.read_item(item)

    views, grants, seen, pins, metadata = [], set(), set(), set(), {}
    cursor, limited, examined = pin, False, 0
    while cursor is not None:
        if cursor in pins:
            raise ValueError('history cycle')
        if len(pins) >= 100 or time.monotonic() - started > 10:
            limited = True
            break
        pins.add(cursor)
        snapshot = memory.snapshot(cursor)
        files = {}
        for path, item in snapshot['files'].items():
            examined += 1
            if examined > 10000 or time.monotonic() - started > 10:
                limited = True
                break
            key = (path, item['sha256'], item.get('saved_ns'))
            if key in seen:
                continue
            seen.add(key)
            files[path] = item
            grants.add((cursor, path))
            metadata[(cursor, path)] = item
        views.append(View(cursor, files))
        if limited or not history:
            break
        cursor = snapshot['parent']
    index = ContentIndex(views, grants, seconds=20, casefold=casefold)
    try:
        result = index.query(query, grants, limit=20)
        result['history_limited'] = limited
        result['scope'] = 'history' if history else 'current'
        if limited:
            result['coverage_complete'] = False
            if not result['snippets']:
                result['status'] = 'INCOMPLETE'
        for snippet in result['snippets']:
            item = metadata[(snippet['version'], snippet['path'])]
            snippet['saved_ns'] = item.get('saved_ns')
            snippet['version_note'] = item.get('version_note', '')
        return result
    finally:
        index.close()
