"""Bounded, session-local exact UTF-8 content index over verified archives.

Derived state only. No filesystem writes, model calls, or permission decisions.
Rebuild when the host changes versions or grants. Not a persistent scale index.
"""
import hashlib
import sqlite3
import time

from local_memory_bridge import Error


class ContentIndex:
    def __init__(self, views, grants, *, source_budget=32 * 1024**2,
                 max_files=10000, sqlite_pages=16384, seconds=30):
        if not (0 < source_budget <= 64 * 1024**2 and 0 < max_files <= 10000
                and 16 <= sqlite_pages <= 16384 and 0 < seconds <= 120):
            raise ValueError("unsupported index budgets")
        views = list(views)
        self._views = {v.version: v for v in views}
        if len(self._views) != len(views):
            raise Error("duplicate archive version")
        self._grants = frozenset(grants)
        if len(self._grants) > max_files or any(
                v not in self._views or p not in self._views[v].files
                for v, p in self._grants):
            raise Error("unknown or excessive scope")
        self._db = sqlite3.connect(":memory:")
        self._rows, self._skipped = {}, []
        self._closed = False
        started = time.monotonic()
        self._db.set_progress_handler(lambda: time.monotonic() - started > seconds, 1000)
        consumed = 0
        try:
            self._db.execute("PRAGMA temp_store=MEMORY")
            self._db.execute("PRAGMA page_size=4096")
            self._db.execute(f"PRAGMA max_page_count={int(sqlite_pages)}")
            self._db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='trigram case_sensitive 1')")
            for version, path in sorted(self._grants):
                if time.monotonic() - started > seconds:
                    raise Error("index build deadline exceeded")
                view = self._views[version]
                item = view.files[path]
                reason = None
                if item['bytes'] > source_budget - consumed:
                    reason = 'source-budget'
                else:
                    consumed += item['bytes']
                    data = view.read(path)
                    if data is None:
                        reason = 'unsupported-codec-or-file-size'
                    else:
                        try:
                            body = data.decode('utf-8')
                        except UnicodeDecodeError:
                            reason = 'not-utf8'
                        else:
                            if '\x00' in body:
                                reason = 'nul-containing-content'
                            else:
                                rowid = len(self._rows) + 1
                                self._db.execute('INSERT INTO docs(rowid,body) VALUES (?,?)', (rowid, body))
                                self._rows[rowid] = (version, path, item['sha256'])
                if reason:
                    self._skipped.append(dict(version=version, path=path, reason=reason))
            self._db.commit()
            self._db.execute('PRAGMA query_only=ON')
            self.build_seconds = time.monotonic() - started
            self.source_bytes_examined = consumed
        except BaseException:
            self.close()
            raise
        finally:
            if not self._closed:
                self._db.set_progress_handler(None, 0)

    def close(self):
        if not self._closed:
            self._db.close()
            self._closed = True

    def query(self, text, grants, *, limit=20, seconds=5):
        """Return exact literal matches; caller must pass its CURRENT grants.

        No-match describes the indexed snapshot, not current disk health.
        Each returned candidate is re-read and cryptographically verified.
        """
        if self._closed:
            raise Error('index closed')
        if frozenset(grants) != self._grants:
            raise Error('scope changed: rebuild index before querying')
        if not isinstance(text, str) or not 3 <= len(text) or len(text.encode('utf-8')) > 256 or '\x00' in text:
            raise ValueError('query needs at least 3 Unicode characters and at most 256 UTF-8 bytes')
        if not 1 <= limit <= 100 or not 0 < seconds <= 30:
            raise ValueError('unsupported query budgets')
        started = time.monotonic()
        self._db.set_progress_handler(lambda: time.monotonic() - started > seconds, 1000)
        snippets = []
        try:
            # Quoting makes FTS operators/punctuation literal, binding prevents SQL injection.
            phrase = '"' + text.replace('"', '""') + '"'
            rows = self._db.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY rowid LIMIT ?',
                                    (phrase, limit + 1)).fetchall()
            for (rowid,) in rows[:limit]:
                if time.monotonic() - started > seconds:
                    raise Error('query deadline exceeded')
                version, path, digest = self._rows[rowid]
                data = self._views[version].read(path)
                if data is None or hashlib.sha256(data).hexdigest() != digest:
                    raise Error('candidate no longer verifies')
                offset = data.find(text.encode('utf-8'))
                if offset < 0:
                    raise Error('index candidate disagrees with verified bytes')
                # Exact quote, not a model paraphrase; byte coordinates refer to original.
                snippets.append(dict(version=version, path=path, sha256=digest,
                                     byte_offset=offset, byte_length=len(text.encode('utf-8')), text=text))
            if time.monotonic() - started > seconds:
                raise Error('query deadline exceeded')
            complete = bool(self._grants) and not self._skipped
            return dict(status='FOUND' if snippets else ('NO_MATCH_IN_INDEXED_SCOPE' if complete else 'INCOMPLETE'),
                        coverage_complete=complete, skipped=list(self._skipped), snippets=snippets,
                        truncated=len(rows) > limit, indexed_files=len(self._rows),
                        semantics='case-sensitive exact UTF-8 substring; pinned archive versions',
                        integrity_scope='returned files reverified; no full archive scrub',
                        content_is_untrusted=True, seconds=time.monotonic() - started)
        finally:
            self._db.set_progress_handler(None, 0)
