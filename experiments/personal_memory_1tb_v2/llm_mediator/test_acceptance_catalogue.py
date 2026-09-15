"""Catalogue validation only: does NOT execute product acceptance scenarios."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance_scenarios import catalogue


class CatalogueTests(unittest.TestCase):
    def test_unique_complete_specifications(self):
        cases = catalogue()['cases']
        self.assertEqual(len(cases), 40)
        self.assertEqual(len({c['id'] for c in cases}), 40)
        for c in cases:
            for key in ('given', 'when', 'oracle'):
                self.assertTrue(c[key].strip())

    def test_no_fabricated_passes(self):
        for c in catalogue()['cases']:
            self.assertEqual(c['status'], 'NOT_EXECUTED')
            self.assertIsNone(c['evidence'])

    def test_traceable_requirement_categories(self):
        for c in catalogue()['cases']:
            self.assertIn(c['basis'], ('user', 'engineering', 'public_case'))


if __name__ == '__main__':
    unittest.main()
