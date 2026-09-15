import unittest
from unittest.mock import patch
from urllib.parse import urlencode
import test_memory_browser as fixture


class ContentUITests(unittest.TestCase):
    setUp = fixture.MemoryBrowserTests.setUp
    stop = fixture.MemoryBrowserTests.stop
    request = fixture.MemoryBrowserTests.request
    upload = fixture.MemoryBrowserTests.upload
    head = fixture.MemoryBrowserTests.head
    download = fixture.MemoryBrowserTests.download

    def test_search_card_download_stays_on_found_version(self):
        self.assertEqual(self.upload('Поиск: оригінал needle'.encode(), self.first)[0], 303)
        found = self.head()
        status, _, body = self.request('GET', path='/token?' + urlencode({'content':'needle'}))
        self.assertEqual(status, 200, body)
        self.assertIn(('snapshot=' + found).encode(), body)
        self.assertIn('Открыть найденную версию'.encode(), body)
        self.assertEqual(self.upload(b'updated content', found)[0], 303)
        status, _, body = self.request('GET', path='/token?' + urlencode({'found':'заметка.txt', 'snapshot':found}))
        self.assertEqual(status, 200)
        self.assertIn(found.encode(), body)
        self.assertEqual(self.download(found, 'заметка.txt')[2], 'Поиск: оригінал needle'.encode())

    def test_short_query_and_unknown_version(self):
        self.assertEqual(self.request('GET', path='/token?content=ab')[0], 400)
        self.assertEqual(self.request('GET', path='/token?found=x&snapshot='+'0'*64)[0], 422)

    def test_html_query_and_file_text_are_escaped(self):
        self.assertEqual(self.upload(b'<script>alert(1)</script>', self.first)[0], 303)
        status, _, body = self.request('GET', path='/token?' + urlencode({'content':'<script>'}))
        self.assertEqual(status, 200, body)
        self.assertNotIn(b'<script>', body)
        self.assertIn(b'&lt;script&gt;', body)

    def test_failed_search_is_not_no_match_and_ui_survives(self):
        with patch.object(fixture.ui, 'worker', side_effect=TimeoutError):
            status, _, body = self.request('GET', path='/token?content=needle')
        self.assertEqual(status, 422)
        self.assertIn('Отсутствие совпадений не установлено'.encode(), body)
        self.assertEqual(self.request('GET')[0], 200)


if __name__ == '__main__':
    unittest.main()
