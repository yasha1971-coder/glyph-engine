import fcntl
from pathlib import Path
import unittest
from unittest.mock import patch
import test_incremental_memory as fixture
import personal_backup as backup
import permanent_delete


class PersonalBackupTests(unittest.TestCase):
    def setUp(self):
        f = fixture.IncrementalMemoryTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        (f.m.root/'browser.lock').touch()
        permanent_delete.set_head(f.m, f.first)

    def test_restore_without_original_directories_and_new_version(self):
        f = self.f
        (f.src/'note.txt').write_bytes(b'Changed document')
        second = f.m.add(f.first, f.src)
        permanent_delete.set_head(f.m,second)
        dest = f.root/'backup'
        pin = backup.create(f.m,dest)
        self.assertEqual(backup.verify(dest,pin)['history_verified']['unique_items_verified'],2)
        f.m.root.rename(f.root/'unavailable-memory')
        f.archive.rename(f.root/'unavailable-archive')
        restored = f.root/'restored'
        result = backup.restore(dest,pin,restored)
        self.assertEqual(result['head'],second)
        m = backup.inc.Memory(restored/'memory',restored/'archive',f.m.base_pin)
        m.restore(f.first,'note.txt',f.root/'old-output')
        self.assertEqual((f.root/'old-output').read_bytes(),b'original\x00bytes')
        m.restore(second,'note.txt',f.root/'new-output')
        self.assertEqual((f.root/'new-output').read_bytes(),b'Changed document')
        (f.src/'note.txt').write_bytes(b'Third version after recovery')
        third=m.add(second,f.src)
        self.assertNotEqual(third,second)

    def test_busy_ui_and_writer_rejected_without_output(self):
        f=self.f
        for name in ('browser.lock','writer.lock'):
            with (f.m.root/name).open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    backup.create(f.m,f.root/'blocked')
                self.assertFalse((f.root/'blocked').exists())

    def test_corruption_and_existing_destination_rejected(self):
        f=self.f; dest=f.root/'backup'
        pin=backup.create(f.m,dest)
        with self.assertRaises(ValueError):
            backup.create(f.m,dest)
        (dest/'memory'/'CURRENT').write_text('bad')
        with self.assertRaises(ValueError):
            backup.restore(dest,pin,f.root/'restore')
        self.assertFalse((f.root/'restore').exists())

    def test_failed_copy_has_no_completion_receipt(self):
        f=self.f; dest=f.root/'failed'
        with patch.object(backup,'copy_file',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                backup.create(f.m,dest)
        self.assertFalse((dest/backup.RECEIPT).exists())
        self.assertEqual(permanent_delete.head(f.m),f.first)

    def test_symlink_pending_and_budget_rejected(self):
        f=self.f
        with self.assertRaises(ValueError):
            backup.create(f.m,f.root/'small',max_bytes=1)
        self.assertFalse((f.root/'small').exists())
        link=f.m.root/'link';link.symlink_to(f.archive)
        with self.assertRaises(ValueError):
            backup.create(f.m,f.root/'link-copy')
        link.unlink()
        (f.m.root/permanent_delete.JOURNAL).write_text('{}')
        with self.assertRaises(ValueError):
            backup.create(f.m,f.root/'pending-copy')


if __name__ == '__main__':
    unittest.main()
