import http.client
from pathlib import Path
import sys
import threading
import unittest
from urllib.parse import urlencode
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory_browser as ui
import incremental_memory as inc
import test_vault_browser as browser_fixture


class MemoryBrowserTests(unittest.TestCase):
    def setUp(self):
        # Reuse archive fixture, not its HTTP server or tests.
        fixture = browser_fixture.VaultBrowserTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.args = fixture.args
        self.args.memory = fixture.root / 'memory'
        self.m = inc.Memory(self.args.memory, self.args.archive, self.args.archive_sha256)
        self.first = self.m.initialize()
        ui.set_head(self.m.root, self.first)
        self.server = ui.HTTPServer(('127.0.0.1', 0), ui.handler_for(self.args, self.m, 'token'))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, body=None, content_type=None, origin=None, path='/token'):
        c = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=20)
        headers = {'Origin': origin or f'http://127.0.0.1:{self.server.server_port}'}
        if content_type:
            headers['Content-Type'] = content_type
        try:
            c.request(method, path, body, headers)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def upload(self, data, pin, path='заметка.txt', origin=None, filename='note.txt'):
        boundary = 'glyph-test-boundary'
        parts = []
        for name, value in [('parent', pin), ('path', path)]:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + data + b'\r\n')
        parts.append(f'--{boundary}--\r\n'.encode())
        return self.request('POST', b''.join(parts), f'multipart/form-data; boundary={boundary}', origin)

    def head(self):
        return (self.m.root / 'CURRENT').read_text().strip()

    def download(self, pin, path):
        return self.request('POST', urlencode({'snapshot': pin, 'path': path}), 'application/x-www-form-urlencoded')

    def test_upload_change_download_both_versions_and_repeat(self):
        self.assertEqual(self.upload(b'first\x00version', self.first)[0], 303)
        second = self.head()
        self.assertEqual(self.upload(b'new version', second)[0], 303)
        third = self.head()
        self.assertEqual(self.download(second, 'заметка.txt')[2], b'first\x00version')
        self.assertEqual(self.download(third, 'заметка.txt')[2], b'new version')
        self.assertEqual(self.upload(b'new version', third)[0], 303)
        self.assertEqual(self.head(), third)
        status, headers, body = self.request('GET')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Referrer-Policy'], 'same-origin')
        self.assertIn(second.encode(), body)
        self.assertIn('заметка.txt'.encode(), body)
        # Reopen storage and CURRENT as after a restart.
        reopened = inc.Memory(self.args.memory, self.args.archive, self.args.archive_sha256)
        self.assertEqual(reopened.snapshot(self.head())['parent'], second)

    def test_stale_tab_does_not_overwrite_current(self):
        self.upload(b'first', self.first)
        current = self.head()
        self.assertEqual(self.upload(b'stale', self.first)[0], 409)
        self.assertEqual(self.head(), current)

    def test_foreign_null_origin_and_traversal_rejected(self):
        for origin in ['https://evil.example', 'null']:
            self.assertEqual(self.upload(b'bad', self.first, origin=origin)[0], 403)
        self.assertEqual(self.upload(b'bad', self.first, path='../escape')[0], 422)
        self.assertEqual(self.head(), self.first)

    def test_corruption_is_not_downloaded(self):
        self.upload(b'payload' * 100, self.first)
        pin = self.head()
        next((self.m.root / 'objects').iterdir()).write_bytes(b'bad')
        status, _, body = self.download(pin, 'заметка.txt')
        self.assertEqual(status, 422)
        self.assertNotIn(b'payload', body)

    def test_oversized_upload_preserves_current(self):
        self.assertEqual(self.upload(b'x' * (inc.LIMIT + 1), self.first)[0], 422)
        self.assertEqual(self.head(), self.first)

    def test_original_unicode_filename_without_custom_path(self):
        self.assertEqual(self.upload(b'example', self.first, path='', filename='файл.txt')[0], 303)
        self.assertEqual(self.download(self.head(), 'файл.txt')[2], b'example')
