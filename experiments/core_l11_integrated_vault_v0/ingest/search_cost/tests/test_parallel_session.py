"""Real file-integrity checks and executor failure/cleanup regression tests."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import parallel_session
from parallel_session import ParallelHashSession
from session import catalog


class CatalogSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.live = self.root / 'live'
        self.expanded = self.root / 'expanded'
        self.live.mkdir()
        self.expanded.mkdir()
        (self.live / 'data').write_bytes(b'abcd')
        (self.expanded / 'data').write_bytes(b'physical state')
        self.state = ParallelHashSession.__new__(ParallelHashSession)
        self.state.vault = self.live
        self.state.expanded = self.expanded
        self.state.states = {}
        self.state.stage = None
        self.state._hash_pool = ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        self.state.close()
        self.directory.cleanup()

    def test_full_catalogs_equal_original_reader(self):
        self.assertEqual(self.state._catalogs(),
                         dict(live=catalog(self.live), expanded=catalog(self.expanded)))

    def test_same_length_and_restored_mtime_does_not_hide_corruption(self):
        original = self.state._catalogs()
        path = self.live / 'data'
        stamp = path.stat()
        path.write_bytes(b'abce')
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertNotEqual(self.state._catalogs(), original)

    def test_new_empty_file_is_not_free_or_ignored(self):
        original = self.state._catalogs()
        (self.live / 'extra').touch()
        self.assertNotEqual(self.state._catalogs(), original)

    def test_symlink_rejected(self):
        (self.live / 'link').symlink_to(self.expanded / 'data')
        with self.assertRaisesRegex(ValueError, 'nonregular session entry'):
            self.state._catalogs()

    def test_sparse_oversized_file_rejected_before_hashing(self):
        with (self.live / 'oversized').open('wb') as output:
            output.truncate((128 << 20) + 1)
        with self.assertRaisesRegex(ValueError, 'session file admission'):
            self.state._catalogs()

    def test_catalog_error_waits_for_other_actual_reads(self):
        (self.live / 'link').symlink_to(self.expanded / 'data')
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        errors = []
        original = parallel_session.catalog

        def observed_read(root):
            if root == self.expanded:
                started.set()
                if not release.wait(5):
                    raise RuntimeError('test coordination timed out')
            return original(root)

        def invoke():
            try:
                self.state._catalogs()
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        # Instrument scheduling only; both workers call the real file reader.
        with patch.object(parallel_session, 'catalog', side_effect=observed_read):
            worker = threading.Thread(target=invoke)
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                self.assertFalse(finished.wait(0.05), 'catalog returned while another read was pending')
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertIn('nonregular session entry', str(errors[0]))

    def test_close_is_idempotent_and_reaps_hash_threads(self):
        self.state._catalogs()
        pool = self.state._hash_pool
        threads = list(pool._threads)
        self.state.close()
        self.state.close()
        self.assertTrue(threads)
        self.assertTrue(all(not thread.is_alive() for thread in threads))


if __name__ == '__main__':
    unittest.main()
