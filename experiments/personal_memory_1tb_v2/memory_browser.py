#!/usr/bin/env python3
"""Local personal-memory UI pilot: one-file upload, versions and verified download."""
import argparse
import email.policy
from datetime import datetime, timezone
from email.parser import BytesParser
import fcntl
import html
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
from http.server import HTTPServer

import incremental_memory as inc
import vault_browser


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


def worker(args, action, pin, extra):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--memory', str(args.memory),
           '--archive', str(args.archive), '--archive-sha256', args.archive_sha256,
           '--worker', action, '--snapshot', pin] + extra
    if args.precompressor:
        cmd += ['--precompressor', str(args.precompressor), '--precompressor-sha256', args.precompressor_sha256]
    with tempfile.TemporaryFile() as output:
        child = subprocess.Popen(cmd, stdout=output, stderr=subprocess.DEVNULL,
                                 preexec_fn=vault_browser.limited_child, start_new_session=True)
        try:
            result = child.wait(timeout=330)
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
    script_nonce = secrets.token_urlsafe(24)
    class Handler(parent):
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

        def current(self):
            pin = inc.read_regular(memory.root, 'CURRENT', 65).decode().strip()
            memory.snapshot(pin)
            return pin

        def history(self):
            pins, pin = [], self.current()
            while pin is not None:
                if pin in pins or len(pins) >= 1000:
                    raise inc.Error('invalid history')
                pins.append(pin)
                pin = memory.snapshot(pin)['parent']
            return pins

        def do_GET(self):
            url = urlsplit(self.path)
            if not self.allowed() or url.path != '/' + token or len(self.path) > 8192:
                return self.send(403, b'Forbidden', 'text/plain')
            try:
                params = parse_qs(url.query)
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
                    page += f'<p>Текущий размер: {files[focus]["bytes"]:,} байт</p><div class="actions">' + download(current, focus)
                    page += f'<a class="button" href="/{token}?update={quote(focus, safe="")}">Обновить этот файл</a></div></section>'
                    changes, last = [], None
                    for pin in reversed(pins):
                        doc = memory.snapshot(pin)
                        item = doc['files'].get(focus)
                        if item and item['sha256'] != last:
                            stamp = item.get('saved_ns', doc.get('created_ns'))
                            date = datetime.fromtimestamp(stamp / 1e9, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if stamp else 'Дата старой записи неизвестна'
                            changes.append((pin, item, date))
                            last = item['sha256']
                        elif item and changes and item.get('filesystem_dates_observed_ms'):
                            # Metadata enrichment is not another content version.
                            old_pin, _, old_date = changes[-1]
                            changes[-1] = (old_pin, item, old_date)
                    page += f'<section class="panel"><h2>История этого файла · {len(changes)}</h2><p>Вверху — последняя сохранённая версия.</p><table>'
                    for number, (pin, item, date) in reversed(list(enumerate(changes, 1))):
                        label = 'Текущая версия' if number == len(changes) else f'Версия {number}'
                        page += f'<tr><td><strong>{label}</strong><br>{esc(item.get("version_note", ""))}<br><small>Сохранено в GLYPH: {date}</small><br><small>Изменено в источнике: {date_value(item.get("source_modified_ms"), 1000)}</small><br>{item["bytes"]:,} байт</td><td>{download(pin, focus, "Скачать эту версию")}</td></tr>'
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
                        page += f'<tr><td><strong>{esc(Path(path).name)}</strong><br><small class="path">{esc(folder)}</small><br><small>{item["bytes"]:,} байт</small></td>'
                        page += f'<td><a class="button" href="/{token}?file={target}">Открыть карточку</a></td></tr>'
                    page += '</table></section>'
                self.send(200, page.encode())
            except Exception:
                self.send(422, 'Состояние памяти не прошло проверку.'.encode())

        def do_POST(self):
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
                    if pin not in self.history():
                        raise inc.Error('unknown version')
                    item = memory.snapshot(pin)['files'][path]
                    with tempfile.TemporaryDirectory(prefix='glyph-download-') as temporary:
                        output = Path(temporary) / 'selected'
                        worker(args, 'restore', pin, ['--path', path, '--output', str(output)])
                        data = inc.read_regular(Path(temporary), 'selected', 64 * 1024**2)
                    if len(data) != item['bytes'] or inc.digest(data) != item['sha256']:
                        raise inc.Error('download verification failed')
                    self.send(200, data, 'application/octet-stream', Path(path).name)
            except Exception:
                self.send(422, 'Операция отклонена. Проверь размер и путь файла. Прежние версии сохранены.'.encode())
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
