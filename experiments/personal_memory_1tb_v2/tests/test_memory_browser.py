import http.client
import re
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

    def upload(self, data, pin, path='заметка.txt', origin=None, filename='note.txt', modified=None, note=None):
        boundary = 'glyph-test-boundary'
        parts = []
        if modified is not None:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="source_modified_ms"\r\n\r\n{modified}\r\n'.encode())
        if note is not None:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="version_note"\r\n\r\n{note}\r\n'.encode())
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
        status, headers, body = self.request('GET', path='/token?file=%D0%B7%D0%B0%D0%BC%D0%B5%D1%82%D0%BA%D0%B0.txt')
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

    def test_duplicate_under_different_name_does_not_add_record(self):
        self.upload(b'hello', self.first, path='', filename='a.txt')
        pin = self.head()
        before = len(self.m.snapshot(pin)['files'])
        status, headers, body = self.upload(b'hello', pin, path='', filename='b.txt')
        self.assertEqual(status, 200)
        self.assertEqual(self.choose(body, '0')[0], 303)
        self.assertEqual(self.head(), pin)
        self.assertEqual(len(self.m.snapshot(pin)['files']), before)

    def test_update_form_binds_original_path_and_new_upload_cannot_overwrite(self):
        self.upload(b'one', self.first, path='', filename='a.txt')
        pin = self.head()
        self.assertEqual(self.upload(b'two', pin, path='', filename='a.txt')[0], 200)
        self.assertEqual(self.head(), pin)
        status, _, page = self.request('GET', path='/token?update=a.txt')
        self.assertEqual(status, 200)
        self.assertIn(b'name="path" value="a.txt"', page)
        self.assertEqual(self.upload(b'two', pin, path='a.txt', filename='different.txt')[0], 303)
        self.assertNotIn('different.txt', self.m.snapshot(self.head())['files'])
        self.assertEqual(self.download(pin, 'a.txt')[2], b'one')

    def test_duplicate_notice_requires_intact_stored_bytes(self):
        self.upload(b'hello', self.first, path='', filename='a.txt')
        pin = self.head()
        next((self.m.root / 'objects').iterdir()).write_bytes(b'bad')
        self.assertEqual(self.choose(self.upload(b'hello', pin, path='', filename='b.txt')[2], '0')[0], 422)
        self.assertEqual(self.head(), pin)

    def test_screens_are_separate_and_save_returns_to_file_card(self):
        status, _, listing = self.request('GET')
        self.assertEqual(status, 200)
        self.assertIn('Открыть карточку'.encode(), listing)
        self.assertNotIn(b'type="file"', listing)
        self.assertNotIn(b'&lt;script&gt;', self.request('GET', path='/token?add=1')[2])
        status, headers, _ = self.upload(b'one', self.first, path='', filename='screen.txt')
        self.assertEqual(status, 303)
        self.assertIn('file=screen.txt', headers['Location'])
        status, _, card = self.request('GET', path=headers['Location'])
        self.assertEqual(status, 200)
        self.assertIn('Карточка файла'.encode(), card)
        self.assertIn('История этого файла'.encode(), card)
        self.assertNotIn(b'type="file"', card)
        self.assertNotIn(b'name="q"', card)
        status, _, update = self.request('GET', path='/token?update=screen.txt')
        self.assertEqual(status, 200)
        self.assertIn(b'name="path" value="screen.txt"', update)
        self.assertIn(b'type="file"', update)
        self.assertNotIn(b'<table>', update)
        self.assertNotIn(b'name="q"', update)
        self.assertNotIn(b'&lt;script&gt;', update)

    def choose(self, body, decision):
        ticket = re.search(rb'name="ticket" value="([^"]+)"', body).group(1).decode()
        return self.request('POST', urlencode({'ticket': ticket, 'decision': decision}), 'application/x-www-form-urlencoded')

    def test_choice_can_keep_separate_without_overwriting_and_reuses_content(self):
        self.upload(b'old', self.first, path='', filename='a.txt')
        pin = self.head()
        body = self.upload(b'new', pin, path='', filename='a.txt')[2]
        self.assertEqual(self.choose(body, 'separate')[0], 303)
        current = self.head()
        self.assertEqual(self.download(current, 'a.txt')[2], b'old')
        self.assertEqual(self.download(current, 'a (2).txt')[2], b'new')
        objects = len(list((self.m.root / 'objects').iterdir()))
        body = self.upload(b'new', current, path='', filename='other.txt')[2]
        self.assertEqual(self.choose(body, 'separate')[0], 303)
        self.assertEqual(len(list((self.m.root / 'objects').iterdir())), objects)

    def test_cancel_and_stale_choice_leave_current(self):
        self.upload(b'old', self.first, path='', filename='a.txt')
        pin = self.head()
        body = self.upload(b'new', pin, path='', filename='a.txt')[2]
        self.assertEqual(self.choose(body, 'cancel')[0], 303)
        self.assertEqual(self.head(), pin)
        body = self.upload(b'new', pin, path='', filename='a.txt')[2]
        self.upload(b'changed', pin, path='a.txt')
        current = self.head()
        self.assertEqual(self.choose(body, '0')[0], 409)
        self.assertEqual(self.head(), current)

    def test_source_date_is_distinct_from_save_date_and_survives_choice(self):
        stamp = 1600000000000
        self.upload(b'old', self.first, path='', filename='a.txt', modified=stamp)
        pin = self.head()
        item = self.m.snapshot(pin)['files']['a.txt']
        self.assertEqual(item['source_modified_ms'], stamp)
        self.assertIsNone(item['source_created_ms'])
        self.assertGreater(item['saved_ns'], stamp * 1000000)
        body = self.upload(b'new', pin, path='', filename='a.txt', modified=stamp + 1000)[2]
        self.assertEqual(self.choose(body, '0')[0], 303)
        self.assertEqual(self.m.snapshot(self.head())['files']['a.txt']['source_modified_ms'], stamp + 1000)
        self.assertEqual(self.m.snapshot(pin)['files']['a.txt'], item)
        page = self.request('GET', path='/token?file=a.txt')[2]
        self.assertIn('Изменён исходный файл'.encode(), page)
        self.assertIn(b'2020-09-13', page)

    def test_invalid_source_date_is_rejected(self):
        self.assertEqual(self.upload(b'x', self.first, modified=-1)[0], 422)
        self.assertEqual(self.head(), self.first)

    def test_filesystem_dates_visible_without_extra_content_version(self):
        import source_dates
        item = self.m.snapshot(self.first)['files']
        name = next(iter(item))
        record = {'path': name, 'status': 'ok', 'bytes': item[name]['bytes'],
                  'sha256': item[name]['sha256'], 'created_ms': 1500000000000,
                  'modified_ms': 1600000000000}
        pin, _ = source_dates.apply(self.m, self.first, [record])
        ui.set_head(self.m.root, pin)
        status, _, page = self.request('GET', path='/token?' + urlencode({'file': name}))
        self.assertEqual(status, 200)
        self.assertIn(b'2017-07-14', page)
        self.assertIn(b'2020-09-13', page)
        self.assertIn('История этого файла · 1'.encode(), page)

    def test_named_versions_restore_and_escape(self):
        self.assertEqual(self.upload(b'first', self.first, note='До исправлений')[0], 303)
        first = self.head()
        self.assertEqual(self.upload(b'second', first, note='<script>sent</script>')[0], 303)
        second = self.head()
        status, _, page = self.request('GET', path='/token?file=' + __import__('urllib.parse', fromlist=['quote']).quote('заметка.txt'))
        self.assertEqual(status, 200)
        self.assertIn('До исправлений', page.decode())
        self.assertIn('&lt;script&gt;sent&lt;/script&gt;', page.decode())
        self.assertEqual(self.download(first, 'заметка.txt')[2], b'first')
        self.assertEqual(self.download(second, 'заметка.txt')[2], b'second')
        self.assertEqual(self.upload(b'third', second, note='x' * 501)[0], 422)
        self.assertEqual(self.head(), second)

    def test_preview_verified_text_pdf_image_and_unsupported(self):
        from urllib.parse import quote
        for name, data, mime in [('sample.txt', b'<script>bad()</script>', 'text/plain'),
                                 ('sample.pdf', b'%PDF-1.4\n%%EOF', 'application/pdf'),
                                 ('sample.png', b'\x89PNG\r\n\x1a\n', 'image/png')]:
            self.assertEqual(self.upload(data, self.head(), path=name)[0], 303)
            status, headers, body = self.request('GET', path='/token?preview=' + name + '&snapshot=' + self.head())
            self.assertEqual(status, 200)
            self.assertEqual(body, data)
            self.assertTrue(headers['Content-Type'].startswith(mime))
            self.assertTrue(headers['Content-Disposition'].startswith('inline;'))
            self.assertIn("default-src 'none'", headers['Content-Security-Policy'])
            if mime != 'application/pdf':
                self.assertIn('sandbox;', headers['Content-Security-Policy'])
            self.assertEqual(headers['Cache-Control'], 'no-store')
        self.upload(b'<svg onload="bad()"/>', self.head(), path='active.svg')
        self.assertEqual(self.request('GET', path='/token?preview=active.svg')[0], 415)

    def test_preview_corruption_and_foreign_origin_are_rejected(self):
        self.upload(b'hello', self.head(), path='preview.txt')
        pin = self.head()
        self.assertEqual(self.request('GET', path='/token?preview=preview.txt', origin='https://other.example')[0], 403)
        (self.m.root / 'objects' / inc.digest(b'hello')).write_bytes(b'broken')
        status, _, data = self.request('GET', path='/token?preview=preview.txt')
        self.assertEqual(status, 422)
        self.assertNotIn(b'hello', data)

    def confirm_delete(self, page):
        ticket = re.search(rb'name="delete_ticket" value="([^"]+)"', page).group(1).decode()
        return self.request('POST', urlencode({'delete_ticket': ticket, 'confirm': 'yes'}), 'application/x-www-form-urlencoded')

    def test_confirmed_deletion_old_links_rejected_and_remaining_downloads(self):
        self.upload(b'one', self.head(), path='versions.txt')
        old = self.head()
        self.upload(b'two', old, path='versions.txt')
        current = self.head()
        status, _, page = self.request('GET', path='/token?delete=versions.txt&version=' + current)
        self.assertEqual(status, 200)
        self.assertEqual(self.head(), current)
        self.assertIn('эту версию'.encode(), page)
        self.assertEqual(self.confirm_delete(page)[0], 303)
        self.assertEqual(self.download(self.head(), 'versions.txt')[2], b'one')
        self.assertEqual(self.download(current, 'versions.txt')[0], 422)
        self.assertEqual(self.request('GET', path='/token?preview=versions.txt&snapshot=' + current)[0], 422)
        page = self.request('GET', path='/token?delete=versions.txt')[2]
        self.assertIn('со всей историей'.encode(), page)
        self.assertEqual(self.confirm_delete(page)[0], 303)
        self.assertNotIn('versions.txt', self.m.snapshot(self.head())['files'])
        self.assertEqual(self.confirm_delete(page)[0], 422)

    def test_stale_delete_and_missing_confirmation_do_not_delete(self):
        self.upload(b'one', self.head(), path='keep.txt')
        page = self.request('GET', path='/token?delete=keep.txt')[2]
        self.upload(b'two', self.head(), path='keep.txt')
        self.assertEqual(self.confirm_delete(page)[0], 409)
        self.assertIn('keep.txt', self.m.snapshot(self.head())['files'])
        self.assertEqual(self.request('POST', urlencode({'delete_ticket': 'madeup', 'confirm': 'yes'}), 'application/x-www-form-urlencoded')[0], 422)

    def test_idle_preconnection_does_not_block_listing(self):
        import socket
        idle = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=2)
        self.addCleanup(idle.close)
        self.assertEqual(self.request('GET')[0], 200)

    def test_broken_connection_does_not_trigger_second_response(self):
        from unittest.mock import patch
        import socket, struct
        finished = threading.Event()
        original = self.server.RequestHandlerClass.preview
        def pause(handler, pin, path):
            finished.wait(2)
            return original(handler, pin, path)
        self.upload(b'plain text', self.head(), path='close.txt')
        with patch.object(self.server, 'handle_error') as error, patch.object(self.server.RequestHandlerClass, 'preview', pause):
            sock = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=2)
            sock.sendall(f'GET /token?preview=close.txt HTTP/1.0\r\nHost: 127.0.0.1:{self.server.server_port}\r\n\r\n'.encode())
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            sock.close()
            finished.set()
            self.assertEqual(self.request('GET')[0], 200)
            # All in-flight preview handlers finish before asserting no traceback.
            for t in list(threading.enumerate()):
                if 'process_request_thread' in t.name:
                    t.join(3)
            error.assert_not_called()

    def test_delete_remains_available_during_slow_preview(self):
        from unittest.mock import patch
        entered, release = threading.Event(), threading.Event()
        self.upload(b'first', self.head(), path='busy.txt')
        old_pin = self.head()
        original = ui.worker
        def slow(*args, **kwargs):
            if args[1] == 'restore':
                entered.set(); release.wait(5)
            return original(*args, **kwargs)
        result = []
        with patch.object(ui, 'worker', slow):
            reader = threading.Thread(target=lambda: result.append(self.request('GET', path='/token?preview=busy.txt')))
            reader.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.request('GET')[0], 200)
                page = self.request('GET', path='/token?delete=busy.txt')[2]
                self.assertEqual(self.confirm_delete(page)[0], 303)
                self.assertNotIn('busy.txt', self.m.snapshot(self.head())['files'])
            finally:
                release.set(); reader.join(5)
        self.assertEqual(result[0][0], 422)
        self.assertNotEqual(self.head(), old_pin)

    def test_corrupted_payload_can_be_deleted_without_preview(self):
        self.upload(b'broken payload', self.head(), path='broken.txt')
        (self.m.root / 'objects' / inc.digest(b'broken payload')).write_bytes(b'bad')
        self.assertEqual(self.request('GET', path='/token?preview=broken.txt')[0], 422)
        page = self.request('GET', path='/token?delete=broken.txt')[2]
        self.assertEqual(self.confirm_delete(page)[0], 303)
        self.assertNotIn('broken.txt', self.m.snapshot(self.head())['files'])

    def test_concurrent_updates_preserve_single_winner(self):
        self.upload(b'start', self.head(), path='race.txt')
        pin = self.head()
        results = []
        threads = [threading.Thread(target=lambda data=d: results.append(self.upload(data, pin, path='race.txt')[0])) for d in (b'one', b'two')]
        for t in threads: t.start()
        for t in threads: t.join(10)
        self.assertEqual(sorted(results), [303, 409])
