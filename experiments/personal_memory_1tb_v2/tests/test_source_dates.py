import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_dates as dates
import test_incremental_memory as fixture


class SourceDatesTests(unittest.TestCase):
    def setUp(self):
        f = fixture.IncrementalMemoryTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.m, self.pin = f.m, f.first
        item = self.m.snapshot(self.pin)['files']['note.txt']
        self.row = {'path': 'note.txt', 'status': 'ok', 'bytes': item['bytes'],
                    'sha256': item['sha256'], 'created_ms': 1500000000000,
                    'modified_ms': 1600000000000}

    def test_matching_metadata_added_without_payload_or_old_snapshot_changes(self):
        old_raw = (self.m.root / 'snapshots' / (self.pin + '.json')).read_bytes()
        pin, report = dates.apply(self.m, self.pin, [self.row])
        self.assertEqual(report['updated'], 1)
        self.assertNotEqual(pin, self.pin)
        item = self.m.snapshot(pin)['files']['note.txt']
        self.assertEqual(item['source_created_ms'], 1500000000000)
        self.assertEqual(item['source_modified_ms'], 1600000000000)
        self.assertNotIn('saved_ns', item)
        self.assertEqual(self.m.read_item(item), b'original\x00bytes')
        self.assertEqual(list((self.m.root / 'objects').iterdir()), [])
        self.assertEqual((self.m.root / 'snapshots' / (self.pin + '.json')).read_bytes(), old_raw)
        again, report = dates.apply(self.m, pin, [self.row])
        self.assertEqual(again, pin)
        self.assertEqual(report['updated'], 0)

    def test_changed_and_missing_sources_are_not_bound(self):
        row = dict(self.row, sha256='0' * 64)
        pin, report = dates.apply(self.m, self.pin, [row])
        self.assertEqual(pin, self.pin)
        self.assertEqual(report['mismatch'], 1)
        pin, report = dates.apply(self.m, self.pin, [{'path': 'note.txt', 'status': 'unavailable'}])
        self.assertEqual(pin, self.pin)
        self.assertEqual(report['unavailable'], 1)

    def test_duplicate_or_invalid_records_publish_nothing(self):
        for rows in ([self.row, self.row], [dict(self.row, created_ms='yesterday')]):
            with self.assertRaises(Exception):
                dates.apply(self.m, self.pin, rows)
        self.assertEqual(len(list((self.m.root / 'snapshots').iterdir())), 1)

    def test_native_request_uses_only_known_paths(self):
        with patch.object(dates, 'collect', return_value=[self.row]) as collect:
            dates.enrich(self.m, self.pin, Path('/mnt/d/example'))
            self.assertEqual(collect.call_args.args[1], [{'path': 'note.txt', 'bytes': self.row['bytes']}])
        self.assertEqual(dates.windows_root('/mnt/d/example'), 'D:\\example')
        with self.assertRaises(Exception):
            dates.windows_root('/home/something')
