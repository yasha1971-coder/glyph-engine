import hashlib
import json
from pathlib import Path
import sys
import unittest
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_content_search import search
import test_memory_browser as browser


class MemoryHistorySearchTests(unittest.TestCase):
    def test_old_version_casefold_and_exact_modes(self):
        class Memory:
            def snapshot(self, pin):
                data = b'Original TRAIN plan' if pin == 'old' else b'New flight plan'
                return {'parent': 'old' if pin == 'new' else None,
                        'files': {'note.txt': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                              'data': data, 'saved_ns': 1 if pin == 'old' else 2,
                                              'version_note': pin}}}
            def read_item(self, item):
                return item['data']
        m = Memory()
        self.assertFalse(search(m, 'new', 'train', casefold=True)['snippets'])
        self.assertFalse(search(m, 'new', 'train', history=True)['snippets'])
        result = search(m, 'new', 'train', history=True, casefold=True)
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['snippets'][0]['version'], 'old')
        self.assertEqual(result['snippets'][0]['text'], 'TRAIN')
        self.assertEqual(result['snippets'][0]['version_note'], 'old')

    def test_history_limit_is_not_complete(self):
        class Memory:
            def snapshot(self, pin):
                return {'parent': str(int(pin) - 1) if int(pin) else None, 'files': {}}
        result = search(Memory(), '110', 'absent', history=True)
        self.assertTrue(result['history_limited'])
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['status'], 'INCOMPLETE')

    def test_history_cycle_rejected(self):
        class Memory:
            def snapshot(self, pin):
                return {'parent': pin, 'files': {}}
        with self.assertRaises(ValueError):
            search(Memory(), 'x', 'text', history=True)

    def test_unchanged_versions_not_duplicated(self):
        class Memory:
            def snapshot(self, pin):
                return {'parent': 'old' if pin == 'new' else None,
                        'files': {'n': {'bytes': 5, 'sha256': hashlib.sha256(b'hello').hexdigest()}}}
            def read_item(self, item):
                return b'hello'
        result = search(Memory(), 'new', 'hello', history=True)
        self.assertEqual(len(result['snippets']), 1)

    def test_browser_upload_history_search_download(self):
        fixture = browser.MemoryBrowserTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        old_bytes = 'Первоначальный ПЛАН поездом'.encode()
        code, _, _ = fixture.upload(old_bytes, fixture.head(), path='history.txt', note='Старый план')
        self.assertEqual(code, 303)
        old_pin = fixture.head()
        code, _, _ = fixture.upload('Теперь самолётом'.encode(), old_pin, path='history.txt')
        self.assertEqual(code, 303)
        query = urlencode({'content':'план', 'scope':'history', 'match':'folded'})
        code, _, page = fixture.request('GET', path='/token?' + query)
        self.assertEqual(code, 200)
        self.assertIn(('snapshot=' + old_pin).encode(), page)
        self.assertIn('Старый план'.encode(), page)
        code, _, page = fixture.request('GET', path='/token?' + urlencode({'found':'history.txt','snapshot':old_pin}))
        self.assertEqual(code, 200)
        code, _, restored = fixture.download(old_pin, 'history.txt')
        self.assertEqual(code, 200)
        self.assertEqual(restored, old_bytes)


if __name__ == '__main__':
    unittest.main()
