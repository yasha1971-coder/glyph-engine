import fcntl
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packed_checkpoint import checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.m = self.root/'m'; self.d = self.root/'d'
        self.m.mkdir(); self.d.mkdir()
        (self.m/'owner.lock').touch(); (self.d/'identity.json').write_text('{"synthetic":true}')
        with sqlite3.connect(self.m/'index.sqlite') as db:
            db.execute('CREATE TABLE versions(id INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO versions VALUES (1)')

    def test_old_database_rejected_despite_unchanged_identity(self):
        old = (self.m/'index.sqlite').read_bytes()
        with sqlite3.connect(self.m/'index.sqlite') as db: db.execute('INSERT INTO versions VALUES (2)')
        pin = checkpoint(self.m,self.d)['checkpoint_sha256']
        self.assertFalse(checkpoint(self.m,self.d,expected=pin)['payload_verified'])
        (self.m/'index.sqlite').write_bytes(old)
        with self.assertRaisesRegex(ValueError,'mismatch'): checkpoint(self.m,self.d,expected=pin)

    def test_busy_and_pending_rejected(self):
        with (self.m/'owner.lock').open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): checkpoint(self.m,self.d)
        for name in ('pending.json','index.sqlite-journal','index.sqlite-wal','index.sqlite-shm'):
            p=self.m/name; p.touch()
            with self.assertRaises(ValueError): checkpoint(self.m,self.d)
            p.unlink()

    def test_symlink_and_identity_change_rejected(self):
        pin = checkpoint(self.m,self.d)['checkpoint_sha256']
        (self.d/'identity.json').write_text('{}')
        with self.assertRaises(ValueError): checkpoint(self.m,self.d,expected=pin)
        (self.m/'index.sqlite').rename(self.m/'other')
        (self.m/'index.sqlite').symlink_to('other')
        with self.assertRaises(OSError): checkpoint(self.m,self.d)
