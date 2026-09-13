#!/usr/bin/env python3
"""Local personal-memory UI pilot: one-file upload, versions and verified download."""
import argparse
import email.policy
from datetime import datetime, timezone
from email.parser import BytesParser
import fcntl
import html
import io
import threading
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, quote, urlsplit
from http.server import ThreadingHTTPServer

import incremental_memory as inc
import vault_browser
import permanent_delete


class HTTPServer(ThreadingHTTPServer):
    """Bound connection threads: browser preconnects cannot occupy the only reader."""
    daemon_threads = True
    block_on_close = False

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(12)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def set_head(root, pin):
    fd, name = tempfile.mkstemp(prefix='.head-', dir=root)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(pin + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, root / 'CURRENT')
        inc.fsync_dir(root)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def worker(args, action, pin, extra, timeout=330):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--memory', str(args.memory),
           '--archive', str(args.archive), '--archive-sha256', args.archive_sha256,
           '--worker', action, '--snapshot', pin] + extra
    if args.precompressor:
        cmd += ['--precompressor', str(args.precompressor), '--precompressor-sha256', args.precompressor_sha256]
    with tempfile.TemporaryFile() as output:
        child = subprocess.Popen(cmd, stdout=output, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        try:
            result = child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
        if result:
            raise inc.Error('worker rejected operation')
        output.seek(0)
        return output.read(4096).decode().strip()


def handler_for(args, memory, token):
    parent = vault_browser.handler_for(args, memory.backend.view, token)
    pending = {}
    deletions = {}
    state_lock = threading.RLock()
    read_slots = threading.BoundedSemaphore(2)
    script_nonce = secrets.token_urlsafe(24)
    class Handler(parent):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def dispatch(self, post=False):
            # Network input/output never owns the state lock. Responses are staged
            # so a disconnect cannot cause a second HTTP response or roll back a write.
            output, input_stream = self.wfile, self.rfile
            buffered = io.BytesIO()
            self.wfile = buffered
            try:
                if not self.allowed() or urlsplit(self.path).path != '/' + token:
                    self.send(403, b'Forbidden', 'text/plain')
                else:
                    readonly = not post and 'preview' in parse_qs(urlsplit(self.path).query)
                    if post:
                        length = int(self.headers.get('Content-Length', '0'))
                        if not 0 < length <= inc.LIMIT + 65536:
                            raise inc.Error('request too large')
                        body = input_stream.read(length)
                        if len(body) != length:
                            raise inc.Error('incomplete body')
                        self.rfile = io.BytesIO(body)
                        if self.headers.get('Content-Type', '').startswith('application/x-www-form-urlencoded'):
                            fields = parse_qs(body.decode())
                            readonly = set(fields) == {'snapshot', 'path'}
                    with state_lock:
                        permanent_delete.recover(memory)
                    action = self.post_locked if post else self.get_locked
                    if readonly:
                        if read_slots.acquire(blocking=False):
                            try:
                                action()
                            finally:
                                read_slots.release()
                        else:
                            self.send(503, 'Уже открываются два файла. Закрой лишний просмотр и повтори.'.encode())
                    else:
                        with state_lock:
                            action()
                self.connection.settimeout(10)
                output.write(buffered.getvalue())
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                self.close_connection = True
            except Exception:
                # Reply only if no HTTP response has been prepared yet.
                if buffered.tell() == 0:
                    try:
                        self.send(422, 'Запрос не выполнен. Обнови страницу и повтори.'.encode())
                        output.write(buffered.getvalue())
                    except (OSError, ValueError):
                        pass
                self.close_connection = True
            finally:
                self.wfile, self.rfile = output, input_stream
                buffered.close()

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(post=True)

        def send_header(self, keyword, value):
            if keyword.lower() == 'content-security-policy':
                value += "; script-src 'nonce-" + script_nonce + "'"
            super().send_header(keyword, value)

        def finish_upload(self, pin, path, data, modified, note=""):
            with tempfile.TemporaryDirectory(prefix='glyph-incoming-') as temporary:
                source = Path(temporary)
                target = source / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                extra = ['--source', str(source), '--version-note=' + note]
                if modified is not None:
                    extra += ['--source-modified-ms', str(modified)]
                new_pin = worker(args, 'add', pin, extra)
            memory.snapshot(new_pin)
            set_head(memory.root, new_pin)
            self.send_response(303)
            self.send_header('Location', '/' + token + '?notice=' + ('unchanged' if new_pin == pin else 'added') + '&file=' + quote(path, safe=''))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def offer_choice(self, pin, path, data, modified, candidates, note=""):
            now = time.monotonic()
            for key in list(pending):
                if now - pending[key]['time'] > 600:
                    del pending[key]
            if len(pending) >= 2:
                return self.send(409, 'Заверши или отмени предыдущий выбор. Через 10 минут незавершённые загрузки освобождаются.'.encode())
            ticket = secrets.token_urlsafe(24)
            pending[ticket] = dict(pin=pin, path=path, data=data, modified=modified, note=note, candidates=candidates, time=now)
            esc = html.escape
            page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:18px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f5f7fa;color:#182330}button{font:inherit;background:#1558a6;color:white;border:0;border-radius:6px;padding:12px;margin:8px 0;cursor:pointer}p{overflow-wrap:anywhere}</style><h1>Как сохранить файл?</h1>'
            page += '<p>Загружен: <strong>' + esc(path) + '</strong></p><p>Найдены совпадения по имени или содержимому. Само совпадение имени не означает, что это один документ.</p>'
            def form(value, label):
                return (f'<form method="post" action="/{token}"><input type="hidden" name="ticket" value="{ticket}">'
                        f'<button name="decision" value="{value}">{label}</button></form>')
            for i, candidate in enumerate(candidates):
                page += '<hr><p>' + esc(candidate) + '</p>' + form(str(i), 'Сохранить новой версией этого файла')
            page += '<hr>' + form('separate', 'Сохранить отдельным файлом')
            page += '<p>Одинаковые данные будут использовать общее хранение. Старые версии сохраняются.</p>'
            page += form('cancel', 'Отмена')
            self.send(200, page.encode())

        def selected_bytes(self, pin, path, timeout=330):
            with state_lock:
                if pin not in self.history():
                    raise inc.Error('unknown version')
                item = memory.snapshot(pin)['files'][path]
            if item['bytes'] > 64 * 1024**2:
                raise inc.Error('preview size limit')
            with tempfile.TemporaryDirectory(prefix='glyph-view-') as temporary:
                output = Path(temporary) / 'selected'
                worker(args, 'restore', pin, ['--path', path, '--output', str(output)], timeout=timeout)
                data = inc.read_regular(Path(temporary), 'selected', 64 * 1024**2)
            if len(data) != item['bytes'] or inc.digest(data) != item['sha256']:
                raise inc.Error('verification failed')
            with state_lock:
                if pin not in self.history():
                    raise inc.Error('version deleted during read')
            return data

        def preview(self, pin, path):
            data = self.selected_bytes(pin, path, timeout=30)
            suffix = Path(path).suffix.lower()
            mime = None
            if suffix == '.pdf' and data.startswith(b'%PDF-'):
                mime = 'application/pdf'
            elif suffix == '.png' and data.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif suffix in ('.jpg', '.jpeg') and data.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            elif suffix == '.gif' and data[:6] in (b'GIF87a', b'GIF89a'):
                mime = 'image/gif'
            elif suffix == '.webp' and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
                mime = 'image/webp'
            elif suffix in ('.txt', '.md', '.csv', '.json', '.log', '.py', '.c', '.cpp', '.h', '.tex', '.yaml', '.yml'):
                data.decode('utf-8-sig')  # Reject binary or unsupported encodings.
                if b'\x00' not in data:
                    mime = 'text/plain; charset=utf-8'
            if mime is None:
                return self.send(415, 'Этот формат пока нельзя просмотреть. Вернись в карточку и нажми «Скачать файл».'.encode())
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Disposition', "inline; filename*=UTF-8''" + quote(Path(path).name, safe=''))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            policy = "default-src 'none'; object-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'" if mime == 'application/pdf' else "sandbox; default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'"
            self.send_header('Content-Security-Policy', policy)
            self.end_headers()
            self.wfile.write(data)

        def current(self):
            with state_lock:
                pin = inc.read_regular(memory.root, 'CURRENT', 65).decode().strip()
                memory.snapshot(pin)
                return pin

        def history(self):
            with state_lock:
                pins, pin = [], self.current()
                while pin is not None:
                    if pin in pins or len(pins) >= 1000:
                        raise inc.Error('invalid history')
                    pins.append(pin)
                    pin = memory.snapshot(pin)['parent']
                return pins

        def get_locked(self):
            url = urlsplit(self.path)
            if not self.allowed() or url.path != '/' + token or len(self.path) > 8192:
                return self.send(403, b'Forbidden', 'text/plain')
            try:
                params = parse_qs(url.query)
                if 'preview' in params:
                    return self.preview(params.get('snapshot', [self.current()])[0], params['preview'][0])
                if 'delete' in params:
                    path = params['delete'][0]
                    current = self.current()
                    selected = params.get('version', [None])[0]
                    if path not in memory.snapshot(current)['files'] or (selected and selected not in self.history()):
                        raise inc.Error('unknown deletion target')
                    meta = memory.snapshot(selected or current)['files'][path]
                    now = time.monotonic()
                    for key in list(deletions):
                        if now - deletions[key]['time'] > 600:
                            del deletions[key]
                    if len(deletions) >= 4:
                        return self.send(409, 'Слишком много открытых подтверждений. Вернись через 10 минут.'.encode())
                    ticket = secrets.token_urlsafe(24)
                    deletions[ticket] = dict(pin=current, path=path, selected=selected, time=now)
                    scope = 'эту версию' if selected else 'документ со всей историей'
                    page = '<!doctype html><meta charset="utf-8"><title>Подтверждение удаления</title><style>body{font:18px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f5f7fa;color:#182330}button{font:inherit;padding:12px;background:#a51b28;color:white;border:0;border-radius:6px}p{overflow-wrap:anywhere}</style><h1>Удалить навсегда ' + scope + '?</h1>'
                    page += '<p>' + html.escape(path) + '</p>'
                    if selected:
                        ordinal, previous_key = 0, None
                        for history_pin, history_doc in permanent_delete.timeline(memory, current):
                            value = history_doc['files'].get(path)
                            key = (value['sha256'], value.get('saved_ns')) if value else None
                            if key is not None and key != previous_key:
                                ordinal += 1
                            previous_key = key
                            if history_pin == selected:
                                break
                        page += '<p>Версия ' + str(ordinal) + ' · ' + str(meta['bytes']) + ' байт</p>'
                        stamp = meta.get('saved_ns')
                        date = datetime.fromtimestamp(stamp / 1e9, timezone.utc).isoformat() if stamp else 'Дата старой версии неизвестна'
                        page += '<p>' + html.escape(date + ' · ' + meta.get('version_note', '')) + '</p>'
                        page += '<p>При удалении текущей версии откроется предыдущая оставшаяся. Если версий больше нет, документ исчезнет из памяти.</p>'
                    page += '<p>Отменить удаление в этой памяти нельзя. Исходные файлы, прежний архив и отдельные резервные копии останутся. Общие данные других версий сохраняются.</p>'
                    page += f'<form method="post" action="/{token}"><input type="hidden" name="delete_ticket" value="{ticket}"><button name="confirm" value="yes">Да, удалить навсегда</button></form>'
                    page += f'<p><a href="/{token}?file={quote(path, safe="")}">Отмена — вернуться в карточку</a></p>'
                    return self.send(200, page.encode())
                pins = self.history() if params.get('file') else [self.current()]
                current = pins[0]
                files = memory.snapshot(current)['files']
                query = params.get('q', [''])[0][:200]
                focus = params.get('file', [''])[0]
                update = params.get('update', [''])[0]
                if (focus and focus not in files) or (update and update not in files):
                    raise inc.Error('unknown file')
                adding = params.get('add', [''])[0] == '1'
                if sum(bool(x) for x in (focus, update, adding)) > 1:
                    raise inc.Error('ambiguous screen')
                esc = html.escape
                page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GLYPH · Мои файлы</title>'
                page += '<style>body{font:17px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182330}h1{font-size:28px}h2{font-size:23px;overflow-wrap:anywhere}.panel{background:white;padding:24px;margin:20px 0;border:1px solid #dae0e8;border-radius:12px}.path{color:#536170;overflow-wrap:anywhere}button,input,.button{font:inherit;padding:10px 14px}button,.button{background:#1558a6;color:white;border:0;border-radius:6px;cursor:pointer;text-decoration:none;display:inline-block}a{color:#1558a6}form{margin:16px 0}table{width:100%;border-collapse:collapse}td{padding:14px 8px;border-bottom:1px solid #e2e6ec;overflow-wrap:anywhere}td:first-child{width:65%}.notice{padding:16px;background:#e4f3e9}.actions{display:flex;gap:16px;align-items:center;flex-wrap:wrap}small{color:#536170}</style>'
                page += '<h1>GLYPH · Личная память</h1>'
                notice = params.get('notice', [''])[0]
                notices = {'added': 'Файл сохранён. Ты находишься в его карточке.',
                           'unchanged': 'Содержимое не изменилось. Новая версия не создана.',
                           'deleted': 'Удаление завершено.',
                           'duplicate': 'Этот файл уже сохранён. Открыта его карточка; лишняя запись не добавлена.'}
                if notice in notices:
                    page += '<p class="notice">' + notices[notice] + '</p>'
                def identity(path):
                    parent_path = str(Path(path).parent)
                    location = 'В корне памяти' if parent_path == '.' else 'Папка: ' + parent_path
                    return '<h2>' + esc(Path(path).name) + '</h2><p class="path">' + esc(location) + '</p>'
                def download(pin, path, label='Скачать файл'):
                    return (f'<form method="post" action="/{token}"><input type="hidden" name="snapshot" value="{pin}">'
                            f'<input type="hidden" name="path" value="{esc(path, quote=True)}"><button>{label}</button></form>')
                def view_link(pin, path, label='Просмотреть'):
                    return f'<a class="button" target="_blank" rel="noopener noreferrer" href="/{token}?preview={quote(path, safe="")}&amp;snapshot={pin}">{label}</a>'
                def delete_link(path, pin=None):
                    target = '/' + token + '?delete=' + quote(path, safe='')
                    if pin:
                        target += '&amp;version=' + pin
                    return f'<p><a href="{target}">Удалить {"эту версию" if pin else "документ"} навсегда</a></p>'
                if update or adding:
                    back = '/' + token + ('?file=' + quote(update, safe='') if update else '')
                    page += f'<p><a href="{back}">← ' + ('В карточку файла' if update else 'Мои файлы') + '</a></p>'
                    page += '<section class="panel"><h2>' + ('Обновить этот файл' if update else 'Добавить новый файл') + '</h2>'
                    if update:
                        page += identity(update)
                        page += '<p>Выбери изменённый файл на ноутбуке. Он станет новой версией этой записи; прежняя сохранится.</p>'
                    else:
                        page += '<p>Выбери файл на ноутбуке, который хочешь сохранить в личной памяти.</p>'
                    page += f'<form method="post" enctype="multipart/form-data" action="/{token}">'
                    page += f'<input type="hidden" name="parent" value="{current}"><input type="hidden" name="path" value="{esc(update, quote=True)}">'
                    page += '<p><label>1. Выбери файл<br><input type="file" id="upload-file" name="file" required><input type="hidden" id="source-modified" name="source_modified_ms" value=""></label></p><p><small>До 8 MiB. Исходный файл на ноутбуке остаётся на месте.</small></p>'
                    page += f'<script nonce="{script_nonce}">' + "document.getElementById('upload-file').addEventListener('change', function(){ const f=this.files[0]; document.getElementById('source-modified').value=f && Number.isFinite(f.lastModified) ? String(f.lastModified) : ''; });" + '</script>'
                    page += '<p><label>Название или заметка к версии (необязательно)<br><input name="version_note" maxlength="500" placeholder="Например: отправлено, до исправлений"></label></p>'
                    page += '<button>2. ' + ('Сохранить новую версию' if update else 'Сохранить в память') + '</button></form></section>'
                elif focus:
                    page += f'<p><a href="/{token}">← Мои файлы</a></p><section class="panel"><p><small>Карточка файла</small></p>'
                    page += identity(focus)
                    def date_value(value, scale):
                        return datetime.fromtimestamp(value / scale, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if value is not None else 'Неизвестно'
                    meta = files[focus]
                    page += '<p>Создан исходный файл: ' + date_value(meta.get('source_created_ms'), 1000) + '</p>'
                    page += '<p>Изменён исходный файл: ' + date_value(meta.get('source_modified_ms'), 1000) + '</p>'
                    if meta.get('filesystem_dates_observed_ms'):
                        page += '<p><small>Даты дополнены из Windows после совпадения SHA-256. Это сведения файловой системы, не доказательство первого появления документа.</small></p>'
                    elif meta.get('source_modified_ms') is not None:
                        page += '<p><small>Дата изменения сообщена браузером.</small></p>'
                    page += '<p>' + esc(meta.get('version_note', '')) + '</p>'
                    page += '<p>Эта версия сохранена в GLYPH: ' + date_value(meta.get('saved_ns'), 1e9) + '</p>'
                    page += f'<p>Текущий размер: {files[focus]["bytes"]:,} байт</p><div class="actions">' + view_link(current, focus) + download(current, focus)
                    page += f'<a class="button" href="/{token}?update={quote(focus, safe="")}">Обновить этот файл</a></div>' + delete_link(focus) + '</section>'
                    changes, last = [], None
                    for pin in reversed(pins):
                        doc = memory.snapshot(pin)
                        item = doc['files'].get(focus)
                        if item and (item['sha256'], item.get('saved_ns')) != last:
                            stamp = item.get('saved_ns', doc.get('created_ns'))
                            date = datetime.fromtimestamp(stamp / 1e9, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if stamp else 'Дата старой записи неизвестна'
                            changes.append((pin, item, date))
                            last = (item['sha256'], item.get('saved_ns'))
                        elif item and changes and item.get('filesystem_dates_observed_ms'):
                            # Metadata enrichment is not another content version.
                            old_pin, _, old_date = changes[-1]
                            changes[-1] = (old_pin, item, old_date)
                    page += f'<section class="panel"><h2>История этого файла · {len(changes)}</h2><p>Вверху — последняя сохранённая версия.</p><table>'
                    for number, (pin, item, date) in reversed(list(enumerate(changes, 1))):
                        label = 'Текущая версия' if number == len(changes) else f'Версия {number}'
                        page += f'<tr><td><strong>{label}</strong><br>{esc(item.get("version_note", ""))}<br><small>Сохранено в GLYPH: {date}</small><br><small>Изменено в источнике: {date_value(item.get("source_modified_ms"), 1000)}</small><br>{item["bytes"]:,} байт</td><td>{view_link(pin, focus)}{download(pin, focus, "Скачать эту версию")}{delete_link(focus, pin)}</td></tr>'
                    page += '</table></section>'
                else:
                    page += f'<div class="actions"><h2>Мои файлы</h2><a class="button" href="/{token}?add=1">Добавить файл</a></div>'
                    page += f'<form action="/{token}"><input name="q" value="{esc(query, quote=True)}" placeholder="Имя файла или папка"><button>Найти</button></form>'
                    matches = [(p, f) for p, f in sorted(files.items()) if query.casefold() in p.casefold()]
                    page += f'<p>Всего: {len(files)} · Найдено: {len(matches)} · Показано: {min(200,len(matches))}</p><section class="panel"><table>'
                    for path, item in matches[:200]:
                        target = quote(path, safe='')
                        folder = str(Path(path).parent)
                        folder = 'В корне памяти' if folder == '.' else 'Папка: ' + folder
                        page += f'<tr><td><strong><a target="_blank" rel="noopener noreferrer" href="/{token}?preview={target}&amp;snapshot={current}">{esc(Path(path).name)}</a></strong><br><small class="path">{esc(folder)}</small><br><small>{item["bytes"]:,} байт</small></td>'
                        page += f'<td><a class="button" href="/{token}?file={target}">Открыть карточку</a></td></tr>'
                    page += '</table></section>'
                self.send(200, page.encode())
            except Exception:
                message = 'Не удалось просмотреть эту версию. Она могла быть удалена, повреждена или превысить лимит чтения. Вернись в карточку: удаление не требует просмотра.' if 'preview' in parse_qs(url.query) else 'Не удалось прочитать состояние памяти.'
                self.send(422, message.encode())

        def post_locked(self):
            if not self.allowed() or self.path != '/' + token:
                return self.send(403, b'Forbidden', 'text/plain')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= inc.LIMIT + 65536:
                    raise inc.Error('request too large')
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise inc.Error('incomplete request')
                ctype = self.headers.get('Content-Type', '')
                if ctype.startswith('multipart/form-data;'):
                    message = BytesParser(policy=email.policy.default).parsebytes(
                        ('Content-Type: ' + ctype + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
                    if not message.is_multipart() or message.defects:
                        raise inc.Error('bad upload')
                    parts = {}
                    for part in message.iter_parts():
                        name = part.get_param('name', header='content-disposition')
                        if name not in ('file', 'parent', 'path', 'source_modified_ms', 'version_note') or name in parts or part.is_multipart():
                            raise inc.Error('bad upload fields')
                        parts[name] = part
                    pin = parts['parent'].get_payload(decode=True).decode()
                    if pin != self.current():
                        return self.send(409, 'Память уже обновилась. Вернись к списку, обнови страницу и повтори добавление.'.encode())
                    filename = parts['file'].get_filename()
                    chosen = parts['path'].get_payload(decode=True).decode().strip() if 'path' in parts else ''
                    raw_modified = parts['source_modified_ms'].get_payload(decode=True).decode() if 'source_modified_ms' in parts else ''
                    modified = int(raw_modified) if raw_modified else None
                    if modified is not None and not 0 <= modified <= 253402300799000:
                        raise inc.Error('invalid source date')
                    note = parts['version_note'].get_payload(decode=True).decode('utf-8') if 'version_note' in parts else ''
                    if len(note) > 500 or '\x00' in note:
                        raise inc.Error('invalid version note')
                    path = inc.valid_path(chosen or filename)
                    data = parts['file'].get_payload(decode=True)
                    if len(data) > inc.LIMIT or len(path) > 1024:
                        raise inc.Error('upload budget exceeded')
                    known = memory.snapshot(pin)['files']
                    if not chosen:
                        candidates = [p for p, item in known.items()
                                      if Path(p).name.casefold() == Path(path).name.casefold() or item['sha256'] == inc.digest(data)]
                        if candidates:
                            # Bound the chooser; no candidate is auto-selected.
                            return self.offer_choice(pin, path, data, modified, candidates[:20], note)
                    return self.finish_upload(pin, path, data, modified, note)

                else:
                    if length > 8192:
                        raise inc.Error('selection too large')
                    fields = parse_qs(body.decode())
                    if 'delete_ticket' in fields:
                        entry = deletions.pop(fields['delete_ticket'][0], None)
                        if entry is None or time.monotonic() - entry['time'] > 600 or fields.get('confirm') != ['yes']:
                            raise inc.Error('confirmation required')
                        if entry['pin'] != self.current():
                            return self.send(409, 'Память изменилась. Открой новое подтверждение удаления.'.encode())
                        new_pin = permanent_delete.delete(memory, entry['pin'], entry['path'], entry['selected'])
                        pending.clear(); deletions.clear()
                        location = '/' + token + '?notice=deleted'
                        if entry['path'] in memory.snapshot(new_pin)['files']:
                            location += '&file=' + quote(entry['path'], safe='')
                        self.send_response(303)
                        self.send_header('Location', location)
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return
                    if 'ticket' in fields:
                        ticket = fields['ticket'][0]
                        entry = pending.get(ticket)
                        if entry is None or time.monotonic() - entry['time'] > 600:
                            pending.pop(ticket, None)
                            return self.send(409, 'Выбор истёк. Выбери файл заново.'.encode())
                        decision = fields.get('decision', [''])[0]
                        if decision == 'cancel':
                            del pending[ticket]
                            self.send_response(303)
                            self.send_header('Location', '/' + token)
                            self.send_header('Content-Length', '0')
                            self.end_headers()
                            return
                        if entry['pin'] != self.current():
                            del pending[ticket]
                            return self.send(409, 'Память обновилась. Выбери файл заново.'.encode())
                        if decision == 'separate':
                            path = entry['path']
                            known = memory.snapshot(entry['pin'])['files']
                            n = 2
                            original = Path(path)
                            while path in known:
                                path = str(original.with_name(original.stem + f' ({n})' + original.suffix))
                                n += 1
                        else:
                            index = int(decision)
                            if not 0 <= index < len(entry['candidates']):
                                raise inc.Error('invalid choice')
                            path = entry['candidates'][index]
                        del pending[ticket]
                        return self.finish_upload(entry['pin'], path, entry['data'], entry['modified'], entry.get('note', ''))
                    pin, path = fields['snapshot'][0], fields['path'][0]
                    data = self.selected_bytes(pin, path)
                    self.send(200, data, 'application/octet-stream', Path(path).name)
            except Exception:
                self.send(422, 'Операция не завершена. Обнови страницу; если ошибка повторяется, сохрани вывод терминала. Не повторяй удаление вслепую.'.encode())
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('memory', 'archive'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--precompressor', type=Path)
    p.add_argument('--precompressor-sha256')
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--enrich-source-dates', type=Path)
    p.add_argument('--worker', choices=['add', 'restore'])
    p.add_argument('--snapshot')
    p.add_argument('--source', type=Path)
    p.add_argument('--source-modified-ms', type=int)
    p.add_argument('--version-note', default='')
    p.add_argument('--path')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.worker:
        vault_browser.limited_child()
    m = inc.Memory(a.memory, a.archive, a.archive_sha256, a.precompressor, a.precompressor_sha256)
    if a.worker:
        if a.worker == 'add':
            print(m.add(a.snapshot, a.source, a.source_modified_ms, a.version_note))
        else:
            m.restore(a.snapshot, a.path, a.output)
        return
    if not m.root.exists():
        pin = m.initialize()
        set_head(m.root, pin)
    # One UI instance owns CURRENT for its lifetime; CLI snapshots cannot move it.
    with (m.root / 'browser.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        permanent_delete.recover(m)
        pin = inc.read_regular(m.root, 'CURRENT', 65).decode().strip()
        m.snapshot(pin)
        if a.enrich_source_dates:
            import source_dates
            print('Читаю даты Windows и проверяю SHA-256 известных файлов. Архив не пересжимается.', flush=True)
            new_pin, report = source_dates.enrich(m, pin, a.enrich_source_dates)
            if new_pin != pin:
                set_head(m.root, new_pin)
            print(json.dumps({'source_dates': report}, ensure_ascii=False), flush=True)
        token = secrets.token_urlsafe(32)
        server = HTTPServer(('127.0.0.1', a.port), handler_for(a, m, token))
        print(f'Открой на ноутбуке: http://127.0.0.1:{server.server_port}/{token}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
