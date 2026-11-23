import tempfile
import unittest
from pathlib import Path
from storage_budget import assess


class Budget(unittest.TestCase):
    def test_sidecars_and_history_can_exhaust_payload_saving(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'payload').write_bytes(b'x'*60)
            self.assertTrue(assess(p, 100)['storage_budget_pass'])
            (p/'history').mkdir()
            (p/'history'/'index').write_bytes(b'x'*41)
            r = assess(p, 100)
            self.assertEqual(r['retained_bytes'], 101)
            self.assertEqual(r['status'], 'BUDGET_REJECTED')

    def test_equal_is_allowed_but_does_not_prove_restore(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'raw').write_bytes(b'x'*100)
            r = assess(p, 100)
            self.assertTrue(r['storage_budget_pass'])
            self.assertFalse(r['byte_perfect_replay_established'])

    def test_symlink_and_empty_reject(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with self.assertRaises(ValueError): assess(p, 100)
            (p/'escape').symlink_to('/etc/hosts')
            with self.assertRaises(ValueError): assess(p, 100)


if __name__ == '__main__':
    unittest.main()
