import fcntl
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import packed_backup as backup


class BackupTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        self.r=Path(t.name); self.m=self.r/'meta'; self.d=self.r/'data'; self.out=self.r/'copy'
        self.m.mkdir(); (self.m/'owner.lock').touch(); (self.m/'index.sqlite').write_bytes(b'synthetic catalogue')
        (self.d/'packs').mkdir(parents=True); (self.d/'identity.json').write_text('{}')
        (self.d/'packs'/'one.pack').write_bytes(b'synthetic payload')

    def test_exact_copy_and_no_overwrite(self):
        pin=backup.create_backup(self.m,self.d,self.out)
        self.assertTrue(backup.verify_backup(self.out,pin)['copy_readback_verified'])
        self.assertEqual((self.out/'data/packs/one.pack').read_bytes(), b'synthetic payload')
        with self.assertRaises(ValueError): backup.create_backup(self.m,self.d,self.out)

    def test_tampered_copy_or_pin_rejected(self):
        pin=backup.create_backup(self.m,self.d,self.out)
        with self.assertRaises(ValueError): backup.verify_backup(self.out,'0'*64)
        (self.out/'metadata/index.sqlite').write_bytes(b'older')
        with self.assertRaises(ValueError): backup.verify_backup(self.out,pin)

    def test_busy_pending_symlink_budget(self):
        with (self.m/'owner.lock').open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): backup.create_backup(self.m,self.d,self.out)
        with self.assertRaises(ValueError): backup.create_backup(self.m,self.d,self.out,max_bytes=1)
        (self.m/'pending.json').touch()
        with self.assertRaises(ValueError): backup.create_backup(self.m,self.d,self.out)
        (self.m/'pending.json').unlink()
        (self.d/'packs'/'link').symlink_to(self.m/'index.sqlite')
        with self.assertRaises(ValueError): backup.create_backup(self.m,self.d,self.out)
        self.assertFalse(self.out.exists())

    def test_failed_readback_never_publishes_completion(self):
        with patch.object(backup,'_hash_regular',return_value='bad'):
            with self.assertRaises(ValueError): backup.create_backup(self.m,self.d,self.out)
        self.assertFalse((self.out/backup.RECEIPT).exists())
        self.assertEqual((self.d/'packs/one.pack').read_bytes(),b'synthetic payload')

    @unittest.skipUnless(importlib.util.find_spec('packed_scale_store'), 'experimental packed backend not installed')
    def test_real_store_restore_without_original_paths(self):
        from packed_scale_store import Store
        m,d,b=self.r/'real-meta',self.r/'real-data',self.r/'real-backup'
        with Store(m,d,create=True) as s:
            first=s.put_version('document',[b'original '*1000])
            second=s.put_version('document',[b'original '*1000,b'changed'])
        pin=backup.create_backup(m,d,b)
        backup.verify_backup(b,pin)
        m.rename(self.r/'unavailable-meta');d.rename(self.r/'unavailable-data')
        with Store(b/'metadata',b/'data') as s:
            self.assertEqual(b''.join(s.read_version(first)),b'original '*1000)
            self.assertEqual(b''.join(s.read_version(second)),b'original '*1000+b'changed')
            self.assertEqual(len(s.history('document')),2)
