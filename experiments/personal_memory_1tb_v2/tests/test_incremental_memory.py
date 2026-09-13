import contextlib
import fcntl
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import incremental_memory as inc
import verified_hybrid_archive as base
from test_verified_hybrid_archive import make_inventory


class IncrementalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        src = self.root / 'original'
        src.mkdir()
        (src / 'note.txt').write_bytes(b'original\x00bytes')
        inv, archive = self.root / 'inventory', self.root / 'archive'
        make_inventory(src, inv)
        with contextlib.redirect_stderr(io.StringIO()):
            base.build(inv, archive, 0)
        self.archive = archive
        self.archive_before = {p.relative_to(archive): p.read_bytes() for p in archive.rglob('*') if p.is_file()}
        self.m = inc.Memory(self.root / 'memory', archive, inc.digest((archive / base.RECEIPT).read_bytes()))
        self.first = self.m.initialize()
        self.src = self.root / 'incoming'
        self.src.mkdir()

    def test_versions_repeat_dedup_revert_and_base_unchanged(self):
        (self.src / 'note.txt').write_bytes(b'new bytes' * 100)
        (self.src / 'copy.txt').write_bytes(b'new bytes' * 100)
        second = self.m.add(self.first, self.src)
        self.assertNotEqual(second, self.first)
        self.assertEqual(len(list((self.m.root / 'objects').iterdir())), 1)
        self.assertEqual(self.m.add(second, self.src), second)
        self.m.restore(self.first, 'note.txt', self.root / 'old')
        self.m.restore(second, 'copy.txt', self.root / 'new')
        self.assertEqual((self.root / 'old').read_bytes(), b'original\x00bytes')
        self.assertEqual((self.root / 'new').read_bytes(), b'new bytes' * 100)
        (self.src / 'note.txt').write_bytes(b'original\x00bytes')
        third = self.m.add(second, self.src)
        self.assertEqual(self.m.snapshot(third)['files']['note.txt']['storage'], 'base')
        self.assertEqual(len(list((self.m.root / 'objects').iterdir())), 1)
        self.assertEqual(self.archive_before, {p.relative_to(self.archive): p.read_bytes()
                         for p in self.archive.rglob('*') if p.is_file()})

    def test_absent_files_are_retained(self):
        (self.src / 'new.txt').write_bytes(b'another')
        pin = self.m.add(self.first, self.src)
        self.assertEqual(set(self.m.snapshot(pin)['files']), {'note.txt', 'new.txt'})

    def test_corrupt_snapshot_is_rejected(self):
        (self.m.root / 'snapshots' / (self.first + '.json')).write_bytes(b'{}')
        with self.assertRaises(inc.Error):
            self.m.add(self.first, self.src)

    def test_corrupt_object_no_output_and_no_successful_reuse(self):
        (self.src / 'new').write_bytes(b'hello' * 100)
        pin = self.m.add(self.first, self.src)
        obj = next((self.m.root / 'objects').iterdir())
        obj.write_bytes(b'corrupt')
        with self.assertRaises(Exception):
            self.m.restore(pin, 'new', self.root / 'restore')
        self.assertFalse((self.root / 'restore').exists())
        with self.assertRaises(Exception):
            self.m.add(pin, self.src)

    def test_failure_before_publication_leaves_old_snapshot_usable(self):
        (self.src / 'new').write_bytes(b'new content')
        with patch.object(self.m, 'commit', side_effect=OSError('simulated failure')):
            with self.assertRaises(OSError):
                self.m.add(self.first, self.src)
        self.assertEqual(len(list((self.m.root / 'snapshots').iterdir())), 1)
        self.m.restore(self.first, 'note.txt', self.root / 'old')
        pin = self.m.add(self.first, self.src)
        self.assertIn('new', self.m.snapshot(pin)['files'])

    def test_busy_writer_rejected(self):
        with (self.m.root / 'writer.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(inc.Error):
                self.m.add(self.first, self.src)

    def test_symlink_and_oversized_input_rejected(self):
        (self.src / 'link').symlink_to(self.root / 'original' / 'note.txt')
        with self.assertRaises(inc.Error):
            self.m.add(self.first, self.src)
        (self.src / 'link').unlink()
        with (self.src / 'large').open('wb') as f:
            f.truncate(inc.LIMIT + 1)
        with self.assertRaises(inc.Error):
            self.m.add(self.first, self.src)

    def test_existing_destination_and_overlapping_source_rejected(self):
        output = self.root / 'out'
        output.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            self.m.restore(self.first, 'note.txt', output)
        self.assertEqual(output.read_bytes(), b'keep')
        with self.assertRaises(inc.Error):
            self.m.add(self.first, self.m.root)
