#!/usr/bin/env python3
"""Local archive browser pilot. Metadata search and explicit verified download.

No ingestion, semantic search, model invocation or full-Vault size claim.
Linux/WSL only; decoder child has resource limits, not a security sandbox.
"""
import argparse
import hashlib
import html
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from compressed_preservation import CompressedPreservation


def limited_child():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    os.umask(0o077)


def restore_one(args, index):
    """Index is an explicit human request; child reopens pinned archive metadata."""
    with tempfile.TemporaryDirectory(prefix="glyph-selected-") as raw:
        output = Path(raw) / "selected.bin"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--archive", str(args.archive),
               "--archive-sha256", args.archive_sha256, "--worker-index", str(index), "--worker-output", str(output)]
        if args.precompressor:
            cmd += ["--precompressor", str(args.precompressor), "--precompressor-sha256", args.precompressor_sha256]
        # All child output files (including nested Precomp output) inherit limits.
        with (Path(raw) / "error.log").open("wb") as log:
            child = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log,
                                     preexec_fn=limited_child, start_new_session=True)
            try:
                code = child.wait(timeout=330)
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
        if code or not output.is_file():
            raise ValueError("Восстановление отклонено: проверка, декодер или лимит ресурсов.")
        if output.stat().st_size > 64 * 1024**2:
            raise ValueError("Файл превышает лимит пилота 64 MiB.")
        return output.read_bytes()


def handler_for(args, view, token):
    files = sorted(view.files)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass  # Do not write private names or access tokens to logs.

        def send(self, status, data, content_type="text/html; charset=utf-8", filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(filename, safe=""))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            host = self.headers.get("Host", "")
            valid = f"127.0.0.1:{self.server.server_port}"
            origin = self.headers.get("Origin")
            return host == valid and (origin is None or origin == "http://" + valid)

        def do_GET(self):
            url = urlsplit(self.path)
            if not self.allowed() or url.path != "/" + token or len(self.path) > 8192:
                return self.send(403, b"Forbidden", "text/plain")
            query = parse_qs(url.query).get("q", [""])[0][:200]
            rows = []
            matches = [(i, p) for i, p in enumerate(files) if query.casefold() in p.casefold()]
            for i, p in matches[:200]:
                size = view.files[p]["bytes"]
                rows.append(f'<tr><td>{html.escape(p)}</td><td>{size:,} B</td><td>'
                            f'<form method="post" action="/{token}"><input type="hidden" name="index" value="{i}">'
                            '<button>Восстановить и скачать</button></form></td></tr>')
            page = '<!doctype html><meta charset="utf-8"><title>GLYPH · Мои файлы</title>'
            page += '<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#f6f7f9;color:#16202b}td{padding:14px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}button,input{font:inherit;padding:10px}input[type=search]{width:65%}small{color:#536170}</style>'
            page += '<h1>GLYPH · Мои файлы</h1><p>Локальный пилот: поиск по имени и папке, восстановление оригинала.</p>'
            page += '<p><small>Поиск по содержимому, ежедневное добавление и Qwen ещё не подключены. '
            page += 'Архив не изменяется. Скачивание начинается только после проверки выбранного файла.</small></p>'
            page += f'<form action="/{token}"><input type="search" name="q" value="{html.escape(query, quote=True)}" placeholder="Имя файла или папка"><button>Найти</button></form>'
            page += f'<p>Файлов: {len(files)} · Найдено: {len(matches)} · Показано: {min(200,len(matches))}</p>'
            page += '<table>' + ''.join(rows) + '</table><p><small>Копия появится в папке загрузок браузера. Исходники остаются на месте. Для остановки: Ctrl+C в терминале.</small></p>'
            self.send(200, page.encode())

        def do_POST(self):
            if not self.allowed() or self.path != "/" + token:
                return self.send(403, b"Forbidden", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 256:
                    raise ValueError("Неверный запрос")
                index = int(parse_qs(self.rfile.read(length).decode())["index"][0])
                if not 0 <= index < len(files):
                    raise ValueError("Неверный выбор")
                path = files[index]
                data = restore_one(args, index)
                item = view.files[path]
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Проверка файла не пройдена")
                self.send(200, data, "application/octet-stream", Path(path).name)
            except Exception:
                self.send(422, "Выдача файла отклонена. Возможна ошибка проверки или превышение лимита пилота. Архив не изменён.".encode(), "text/plain; charset=utf-8")
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--precompressor", type=Path)
    p.add_argument("--precompressor-sha256")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--worker-index", type=int)
    p.add_argument("--worker-output", type=Path)
    a = p.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Этот пилот запускается в Linux/WSL.")
    b = CompressedPreservation(a.archive, a.archive_sha256, a.precompressor, a.precompressor_sha256)
    if a.worker_index is not None:
        names = sorted(b.view.files)
        if not 0 <= a.worker_index < len(names) or a.worker_output is None:
            raise ValueError("invalid worker selection")
        name = names[a.worker_index]
        item = b.view.files[name]
        data = b.read_verified(name, item["bytes"], item["sha256"])
        with a.worker_output.open("xb") as stream:
            stream.write(data)
        return
    token = secrets.token_urlsafe(32)
    server = HTTPServer(("127.0.0.1", a.port), handler_for(a, b.view, token))
    print(f"Открой на этом ноутбуке: http://127.0.0.1:{server.server_port}/{token}", flush=True)
    print("Архив открыт по закреплённому хешу. Проверка отдельных файлов — при скачивании. Ctrl+C — остановить.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
