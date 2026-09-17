import fcntl
import io
import unittest
from unittest.mock import patch

import personal_manager as manager
import permanent_delete
import test_incremental_memory as fixture


class ManagerTests(unittest.TestCase):
    def setUp(self):
        f = fixture.IncrementalMemoryTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        (f.m.root / 'browser.lock').touch()
        permanent_delete.set_head(f.m, f.first)
        self.m = manager.Manager(f.root / 'manager')
        p = manager.profile_for(f.m.root, f.archive, f.m.base_pin)
        self.ident = self.m.save_profile(p)
        self.m.activate(self.ident)

    def test_full_menu_backup_restore_switch_and_launch(self):
        f = self.f
        (f.src / 'note.txt').write_bytes(b'new version')
        second = f.m.add(f.first, f.src)
        permanent_delete.set_head(f.m, second)
        with patch('builtins.input', side_effect=['2', str(f.root / 'copy'), '0']), patch('sys.stdout', new_callable=io.StringIO):
            manager.menu(self.m, 1024**3)
        record = manager.read_json(next((self.m.state / 'backups').glob('*.json')))
        restored = f.root / 'recovered'
        new_id = self.m.restore_backup(record['backup'], record['sha256'], restored, None, 1024**3)
        self.assertEqual(self.m.active()['memory'], str(f.m.root))
        # Recovery works with both original directories unavailable.
        f.m.root.rename(f.root / 'old-memory-offline')
        f.archive.rename(f.root / 'old-base-offline')
        self.m.activate(new_id)
        m = manager.memory(self.m.active())
        for pin, data in [(f.first, b'original\x00bytes'), (second, b'new version')]:
            out = f.root / pin
            m.restore(pin, 'note.txt', out)
            self.assertEqual(out.read_bytes(), data)
        with patch.object(manager.subprocess, 'Popen') as launch, patch('sys.stdout', new_callable=io.StringIO):
            launch.return_value.wait.return_value = 0
            self.m.launch()
            args = launch.call_args.args[0]
            self.assertEqual(args[args.index('--memory') + 1], str(restored / 'memory'))
            self.assertEqual(args[1], str(manager.HERE / 'memory_browser.py'))

    def test_failed_restore_keeps_active_and_profiles(self):
        f = self.f
        record = self.m.create_backup(f.root / 'copy', 1024**3)
        (f.root / 'copy/memory/CURRENT').write_text('corrupt')
        with self.assertRaises(ValueError):
            self.m.restore_backup(record['backup'], record['sha256'], f.root / 'bad', None, 1024**3)
        self.assertFalse((f.root / 'bad').exists())
        self.assertEqual(len(list((self.m.state / 'profiles').glob('*'))), 1)
        self.assertEqual(self.m.active()['memory'], str(f.m.root))

    def test_recovery_menu_without_original_state(self):
        f = self.f
        record = self.m.create_backup(f.root / 'copy', 1024**3)
        f.m.root.rename(f.root / 'offline-memory')
        f.archive.rename(f.root / 'offline-archive')
        fresh = manager.Manager(f.root / 'fresh-manager')
        answers = ['3', '2', record['backup'], record['sha256'], '',
                   str(f.root / 'restored'), '4', '1', 'ДА', '0']
        with patch('builtins.input', side_effect=answers), patch('sys.stdout', new_callable=io.StringIO):
            manager.menu(fresh, 1024**3)
        self.assertEqual(fresh.active()['memory'], str(f.root / 'restored/memory'))
        self.assertEqual(permanent_delete.head(manager.memory(fresh.active())), f.first)

    def test_busy_backup_and_second_manager(self):
        f = self.f
        with (f.m.root / 'browser.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.m.create_backup(f.root / 'busy', 1024**3)
        self.assertFalse(list((self.m.state / 'backups').glob('*')))
        with self.m.lock():
            with self.assertRaises(BlockingIOError):
                with manager.Manager(self.m.state).lock():
                    pass

    def test_bad_activation_preserves_previous_selection(self):
        p = self.m.get(self.ident)
        p['archive_sha256'] = '0' * 64
        with self.assertRaises(manager.ArchiveError):
            self.m.save_profile(p)
        with self.assertRaises(ValueError):
            self.m.activate('../escape')
        self.assertEqual(self.m.active(), self.m.get(self.ident))

    def test_manager_cannot_be_inside_data_or_backup(self):
        with self.assertRaises(ValueError):
            self.m.create_backup(self.m.state / 'copy', 1024**3)
        other = manager.Manager(self.f.m.root / 'manager')
        with self.assertRaises(ValueError):
            other.save_profile(self.m.get(self.ident))


if __name__ == '__main__':
    unittest.main()
