"""Read-only search adapter for a pinned incremental-memory snapshot."""
from indexed_memory import ContentIndex


def search(memory, pin, query):
    class View:
        version = pin
        files = memory.snapshot(pin)['files']

        def read(self, path):
            item = self.files[path]
            if item['bytes'] > 8 * 1024**2:
                return None
            return memory.read_item(item)

    view = View()
    grants = {(pin, path) for path in view.files}
    index = ContentIndex([view], grants, seconds=20)
    try:
        return index.query(query, grants, limit=20)
    finally:
        index.close()
