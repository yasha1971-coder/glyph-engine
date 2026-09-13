import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import incremental_memory as inc
import permanent_delete as deletion
import test_incremental_memory as fixture


class PermanentDeleteTests(unittest.TestCase):
    def setUp(self):
        f = fixture.IncrementalMemoryTests(); f.setUp()
        self.addCleanup(f.doCleanups)
        self.f, self.m, self.src = f, f.m, f.src
        deletion.set_head(self.m, f.first)

    def add(self, data, path='note.txt'):
        for p in self.src.iterdir():
            p.unlink()
        (self.src / path).write_bytes(data)
        pin = self.m.add(deletion.head(self.m), self.src)
        deletion.set_head(self.m, pin)
        return pin

    def test_delete_document_removes_all_history_and_exclusive_payload(self):
        pin = self.add(b'one')
        pin = self.add(b'two')
        old_pins = [p for p, _ in deletion.timeline(self.m, pin)]
        result = deletion.delete(self.m, pin, 'note.txt')
        for _, doc in deletion.timeline(self.m, result):
            self.assertNotIn('note.txt', doc['files'])
        for old in old_pins:
            self.assertFalse((self.m.root / 'snapshots' / (old + '.json')).exists())
        self.assertEqual(list((self.m.root / 'objects').iterdir()), [])
        self.assertEqual(self.f.archive_before, {p.relative_to(self.f.archive): p.read_bytes()
                         for p in self.f.archive.rglob('*') if p.is_file()})

    def test_delete_middle_and_latest_preserves_other_versions(self):
        second = self.add(b'two'); third = self.add(b'three')
        result = deletion.delete(self.m, third, 'note.txt', second)
        data = [self.m.read_item(d['files']['note.txt']) for _, d in deletion.timeline(self.m, result)]
        self.assertEqual(data, [b'original\x00bytes', b'original\x00bytes', b'three'])
        result = deletion.delete(self.m, result, 'note.txt', result)
        self.assertEqual(self.m.read_item(self.m.snapshot(result)['files']['note.txt']), b'original\x00bytes')

    def test_shared_object_and_other_document_survive(self):
        first = self.add(b'shared')
        pin = self.add(b'shared', 'other.txt')
        result = deletion.delete(self.m, pin, 'note.txt')
        self.assertEqual(self.m.read_item(self.m.snapshot(result)['files']['other.txt']), b'shared')
        self.assertEqual(len(list((self.m.root / 'objects').iterdir())), 1)

    def test_shared_legacy_ranges_survive_deleted_original_version(self):
        data = random.Random(27).randbytes(512 * 1024)
        first = self.add(data)
        pin = self.add(b'insert' + data)
        self.assertEqual(self.m.snapshot(pin)['files']['note.txt']['storage'], 'chunks-v1')
        result = deletion.delete(self.m, pin, 'note.txt', first)
        self.assertEqual(self.m.read_item(self.m.snapshot(result)['files']['note.txt']), b'insert' + data)
        self.assertTrue((self.m.root / 'objects' / inc.digest(data)).exists())
        result = deletion.delete(self.m, result, 'note.txt')
        for directory in ('objects', 'recipes', 'chunks'):
            self.assertEqual(list((self.m.root / directory).iterdir()), [])

    def test_failure_before_head_is_aborted_on_restart(self):
        pin = self.add(b'two')
        with patch.object(deletion, 'set_head', side_effect=OSError('power loss')):
            with self.assertRaises(OSError):
                deletion.delete(self.m, pin, 'note.txt')
        deletion.recover(self.m)
        self.assertEqual(deletion.head(self.m), pin)
        self.assertEqual(self.m.read_item(self.m.snapshot(pin)['files']['note.txt']), b'two')
        self.assertFalse((self.m.root / deletion.JOURNAL).exists())
        self.assertEqual(len(list((self.m.root / 'snapshots').iterdir())), 2)

    def test_failure_during_cleanup_completes_on_restart(self):
        pin = self.add(b'two')
        actual = deletion.checked_unlink
        count = 0
        def fail_after_one(m, path):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('power loss')
            actual(m, path)
        with patch.object(deletion, 'checked_unlink', side_effect=fail_after_one):
            with self.assertRaises(OSError):
                deletion.delete(self.m, pin, 'note.txt')
        deletion.recover(self.m)
        self.assertNotIn('note.txt', self.m.snapshot(deletion.head(self.m))['files'])
        self.assertEqual(list((self.m.root / 'objects').iterdir()), [])
        self.assertFalse((self.m.root / deletion.JOURNAL).exists())

    def test_stale_and_detached_histories_rejected(self):
        pin = self.add(b'two')
        with self.assertRaises(inc.Error):
            deletion.delete(self.m, self.f.first, 'note.txt')
        self.m.commit(None, {})
        with self.assertRaises(inc.Error):
            deletion.delete(self.m, pin, 'note.txt')
        self.assertEqual(deletion.head(self.m), pin)

    def test_revert_is_distinct_version_after_deletion(self):
        first = self.add(b'A'); middle = self.add(b'B'); last = self.add(b'A')
        result = deletion.delete(self.m, last, 'note.txt', middle)
        chain = deletion.timeline(self.m, result)
        result = deletion.delete(self.m, result, 'note.txt', chain[1][0])
        self.assertEqual(self.m.read_item(self.m.snapshot(result)['files']['note.txt']), b'A')
