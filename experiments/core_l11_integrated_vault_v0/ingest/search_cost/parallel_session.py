"""Opt-in scheduling experiment: unchanged full SHA catalogs, two read workers.

The default Session, receipts and trust boundary are unchanged. Every live and
private file is still read and hashed before and after each query. This retains
the original quiescent-filesystem scope, not a concurrent-mutation guarantee.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
import subprocess

from session import Session, catalog, code_catalog, require


class ParallelHashSession(Session):
    def __init__(self, *args, **kwargs):
        self._hash_pool = None
        super().__init__(*args, **kwargs)
        try:
            self._hash_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix='glyph-full-sha')
        except BaseException:
            super().close()
            raise

    def _catalogs(self):
        roots = [('live', self.vault), ('expanded', self.expanded)]
        roots.extend((key, Path(state.stage.name))
                     for key, state in self.states.items())
        futures = [(key, self._hash_pool.submit(catalog, root))
                   for key, root in roots]
        try:
            return {key: future.result() for key, future in futures}
        finally:
            # On an error, finish every read before close can remove its files.
            wait([future for _, future in futures])

    def _guard(self):
        require(subprocess.check_output(
            ['git', '-C', str(self.reference), 'rev-parse', 'HEAD'],
            text=True).strip() == 'c33cb493bc75f13ee4a904fabb190a29a196c94b',
            'frozen reference HEAD changed; reopen session')
        current = self._catalogs()
        require(current.pop('live') == self.live,
                'live Vault changed; reopen session')
        require(current == self.private,
                'private snapshot changed; reopen session')
        require(code_catalog(self.reference) == self.codes,
                'code changed; reopen session')

    def close(self):
        pool, self._hash_pool = self._hash_pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        super().close()
