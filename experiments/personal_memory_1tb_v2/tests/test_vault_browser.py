import contextlib
import http.client
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vault_browser as browser
import verified_hybrid_archive as base
from test_verified_hybrid_archive import make_inventory


class VaultBrowserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / "source"
        source.mkdir()
        self.name = '<script>привет.txt'
        self.data = 'Личная память\n'.encode() * 100
        (source / self.name).write_bytes(self.data)
        inv, archive = self.root / "inventory", self.root / "archive"
        make_inventory(source, inv)
        with contextlib.redirect_stderr(io.StringIO()):
            base.build(inv, archive, 0)
        self.args = SimpleNamespace(archive=archive, archive_sha256=base.sha256_bytes((archive / base.RECEIPT).read_bytes()),
                                    precompressor=None, precompressor_sha256=None)
        self.backend = browser.CompressedPreservation(archive, self.args.archive_sha256)
        self.server = browser.HTTPServer(('127.0.0.1', 0), browser.handler_for(self.args, self.backend.view, 'token'))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method='GET', path='/token', body=None, headers=None):
        c = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=15)
        try:
            c.request(method, path, body, headers or {})
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def test_browser_escapes_names_and_rejects_foreign_access(self):
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertIn(b'&lt;script&gt;', body)
        self.assertNotIn(b'<script>', body)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(self.request(path='/wrong')[0], 403)
        self.assertEqual(self.request(headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('POST', body='index=0', headers={'Origin': 'https://evil.example'})[0], 403)

    def test_click_downloads_verified_bytes_from_worker(self):
        status, headers, body = self.request('POST', body='index=0')
        self.assertEqual(status, 200)
        self.assertEqual(body, self.data)
        self.assertIn('attachment', headers['Content-Disposition'])

    def test_corruption_returns_no_private_payload(self):
        obj = next(iter(self.backend.view.objects.values()))
        (self.args.archive / obj['path']).write_bytes(b'bad')
        status, headers, body = self.request('POST', body='index=0')
        self.assertEqual(status, 422)
        self.assertNotIn(self.data, body)

    def test_form_policy_preserves_same_origin_post(self):
        status, headers, _ = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(headers['Referrer-Policy'], 'same-origin')
        origin = f'http://127.0.0.1:{self.server.server_port}'
        status, headers, body = self.request(
            'POST', body='index=0', headers={'Origin': origin})
        self.assertEqual(status, 200)
        self.assertEqual(body, self.data)
        self.assertIn('attachment', headers['Content-Disposition'])

    def test_null_origin_is_still_rejected(self):
        self.assertEqual(self.request(
            'POST', body='index=0', headers={'Origin': 'null'})[0], 403)
