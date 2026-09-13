#!/usr/bin/env python3
"""Local personal-memory UI pilot: one-file upload, versions and verified download."""
import argparse
import email.policy
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
    class Handler(parent):
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
                pins = self.history()
                pin = params.get('snapshot', [pins[0]])[0]
                if pin not in pins:
                    raise inc.Error('unknown version')
                query = params.get('q', [''])[0][:200]
                files = memory.snapshot(pin)['files']
                esc = html.escape
                page = '<!doctype html><meta charset="utf-8"><title>GLYPH · Личная память</title>'
                page += '<style>body{font:17px system-ui;max-width:1000px;margin:35px auto;padding:20px}button,input,select{font:inherit;padding:8px}td{padding:10px;overflow-wrap:anywhere}small{color:#555}</style>'
                page += '<h1>GLYPH · Личная память</h1><p>Добавить → найти → вернуть оригинал.</p>'
                page += '<p><small>Отдельный пилот. Один файл до 8 MiB за добавление. Прежний архив остаётся необходимой основой. Поиск — по имени; LLM ещё не подключена.</small></p>'
                page += f'<form method="post" enctype="multipart/form-data" action="/{token}">'
                page += f'<input type="hidden" name="parent" value="{pins[0]}"><input type="file" name="file" required>'
                page += '<p><input name="path" placeholder="Путь в памяти (необязательно)"></p><p><small>Одинаковый путь создаёт новую версию файла. Пустое поле — имя выбранного файла.</small></p><button>Добавить в память</button></form><hr>'
                page += f'<form action="/{token}"><select name="snapshot">'
                for i, version in enumerate(pins):
                    selected = ' selected' if version == pin else ''
                    label = 'Сейчас' if i == 0 else f'Версия {len(pins)-i}'
                    page += f'<option value="{version}"{selected}>{label}</option>'
                page += f'</select> <input name="q" value="{esc(query, quote=True)}" placeholder="Имя или папка"><button>Показать</button></form>'
                matches = [(p, f) for p, f in sorted(files.items()) if query.casefold() in p.casefold()]
                page += f'<p>Файлов: {len(files)} · Найдено: {len(matches)} · Показано: {min(200,len(matches))}</p><table>'
                for path, item in matches[:200]:
                    page += f'<tr><td>{esc(path)}</td><td>{item["bytes"]} B</td><td><form method="post" action="/{token}">'
                    page += f'<input type="hidden" name="snapshot" value="{pin}"><input type="hidden" name="path" value="{esc(path, quote=True)}"><button>Восстановить и скачать</button></form></td></tr>'
                self.send(200, (page + '</table>').encode())
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
                        if name not in ('file', 'parent', 'path') or name in parts or part.is_multipart():
                            raise inc.Error('bad upload fields')
                        parts[name] = part
                    pin = parts['parent'].get_payload(decode=True).decode()
                    if pin != self.current():
                        return self.send(409, 'Память уже обновилась. Вернись к списку, обнови страницу и повтори добавление.'.encode())
                    filename = parts['file'].get_filename()
                    chosen = parts['path'].get_payload(decode=True).decode().strip() if 'path' in parts else ''
                    path = inc.valid_path(chosen or filename)
                    data = parts['file'].get_payload(decode=True)
                    if len(data) > inc.LIMIT or len(path) > 1024:
                        raise inc.Error('upload budget exceeded')
                    with tempfile.TemporaryDirectory(prefix='glyph-incoming-') as temporary:
                        source = Path(temporary)
                        target = source / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        new_pin = worker(args, 'add', pin, ['--source', str(source)])
                    memory.snapshot(new_pin)
                    set_head(memory.root, new_pin)
                    self.send_response(303)
                    self.send_header('Location', '/' + token)
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                else:
                    if length > 8192:
                        raise inc.Error('selection too large')
                    fields = parse_qs(body.decode())
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
    p.add_argument('--worker', choices=['add', 'restore'])
    p.add_argument('--snapshot')
    p.add_argument('--source', type=Path)
    p.add_argument('--path')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    m = inc.Memory(a.memory, a.archive, a.archive_sha256, a.precompressor, a.precompressor_sha256)
    if a.worker:
        if a.worker == 'add':
            print(m.add(a.snapshot, a.source))
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
