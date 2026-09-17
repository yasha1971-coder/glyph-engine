"""Single local entry point; trusted local filesystem and cooperating writers.

Profiles and backup pins are private local state, never repository content.
Restoration does not activate a profile. Code and decoder are independently trusted.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import personal_backup as backup
import permanent_delete
from verified_hybrid_archive import ArchiveError

HERE = Path(__file__).resolve().parent
BASE_PIN = 'e3253b6e1ef269a85f94ec632d581ffffe8df22ed554b2b46e9c86f8adeddf55'
DECODER_PIN = '6a5289ba81ea658e8e7984bf32791635f53b8e7cee4aafc26b853c7cd5b018f8'


def read_json(path):
    path = Path(path)
    return json.loads(backup.inc.read_regular(path.parent, path.name, 1024 * 1024))


def write_new(path, data):
    backup.inc.publish(Path(path), json.dumps(data, ensure_ascii=False, sort_keys=True).encode())


def memory(profile):
    if profile.get('format') != 'GLYPH_PERSONAL_PROFILE_V1':
        raise ValueError('неизвестный формат настройки')
    return backup.inc.Memory(Path(profile['memory']), Path(profile['archive']),
                             profile['archive_sha256'],
                             Path(profile['precompressor']) if profile['precompressor'] else None,
                             profile['precompressor_sha256'])


def profile_for(root, archive, pin, decoder=None, decoder_pin=None):
    profile = dict(format='GLYPH_PERSONAL_PROFILE_V1', memory=str(Path(root).resolve()),
                   archive=str(Path(archive).resolve()), archive_sha256=pin,
                   precompressor=str(Path(decoder).resolve()) if decoder else None,
                   precompressor_sha256=decoder_pin)
    m = memory(profile)  # validate pinned base before persisting a configuration
    if m.root.exists():
        m.snapshot(permanent_delete.head(m))
    return profile


class Manager:
    def __init__(self, state):
        self.state = Path(state).resolve()
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in ('profiles', 'backups'):
            (self.state / name).mkdir(mode=0o700, exist_ok=True)

    @contextmanager
    def lock(self):
        fd = os.open(self.state / 'manager.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def save_profile(self, profile):
        m = memory(profile)
        for root in (m.root, m.backend.root):
            backup.disjoint(root, self.state)
        ident = uuid.uuid4().hex
        write_new(self.state / 'profiles' / (ident + '.json'), profile)
        return ident

    def get(self, ident):
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('неверный идентификатор настройки')
        return read_json(self.state / 'profiles' / (ident + '.json'))

    def active(self):
        return self.get(backup.inc.read_regular(self.state, 'ACTIVE', 64).decode().strip())

    def activate(self, ident):
        profile = self.get(ident)
        m = memory(profile)
        if m.root.exists():
            m.snapshot(permanent_delete.head(m))
        fd, name = tempfile.mkstemp(prefix='.active-', dir=self.state)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(ident.encode())
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.state / 'ACTIVE')
            backup.inc.fsync_dir(self.state)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def launch(self):
        p = self.active()
        memory(p)
        args = [sys.executable, str(HERE / 'memory_browser.py'), '--memory', p['memory'],
                '--archive', p['archive'], '--archive-sha256', p['archive_sha256']]
        if p['precompressor']:
            args += ['--precompressor', p['precompressor'],
                     '--precompressor-sha256', p['precompressor_sha256']]
        print('Откройте ссылку, которую покажет приложение. Ctrl+C закроет его и вернёт меню.', flush=True)
        child = subprocess.Popen(args)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            # Terminal Ctrl+C reaches both processes. Wait for browser workers/locks.
            print('\nЖду завершения приложения…', flush=True)
            code = child.wait()
        if code not in (0, -2, 130):
            raise RuntimeError('приложение завершилось с кодом ' + str(code))

    def create_backup(self, destination, budget):
        destination = Path(destination).resolve()
        backup.disjoint(destination, self.state)
        pin = backup.create(memory(self.active()), destination, max_bytes=budget)
        record = dict(backup=str(destination), sha256=pin)
        write_new(self.state / 'backups' / (uuid.uuid4().hex + '.json'), record)
        return record

    def restore_backup(self, source, pin, destination, decoder, budget):
        destination = Path(destination).resolve()
        backup.disjoint(destination, self.state)
        doc = backup.verify(Path(source), pin)
        backup.restore(Path(source), pin, destination, precompressor=decoder, max_bytes=budget)
        profile = profile_for(destination / 'memory', destination / 'archive',
                              doc['archive_sha256'], decoder, doc['precompressor_sha256'])
        return self.save_profile(profile)  # deliberately leave ACTIVE unchanged


def discover(base):
    receipts = (base / 'GLYPH-V2-runs').glob('*/GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json')
    matches = [p.parent for p in receipts if not p.is_symlink() and
               p.stat().st_size <= 8 * 1024**2 and hashlib.sha256(p.read_bytes()).hexdigest() == BASE_PIN]
    if len(matches) != 1:
        raise ValueError('ожидался один ранее проверенный архив; найдено: ' + str(len(matches)))
    return profile_for(base / 'GLYPH-V2-runs/personal-memory-ui-v1', matches[0], BASE_PIN,
                       base / 'tools/precomp-v0.4.7/linux/precomp', DECODER_PIN)


def choose(items, label):
    if not items:
        raise ValueError('список пуст')
    for number, item in enumerate(items, 1):
        print(str(number) + '. ' + label(item))
    number = int(input('Номер (0 — отмена): '))
    if number == 0:
        return None
    if not 1 <= number <= len(items):
        raise ValueError('нет такого номера')
    return items[number - 1]


def menu(manager, budget):
    while True:
        active = manager.active() if (manager.state / 'ACTIVE').exists() else None
        print('\nGLYPH — личная память\nРабочая папка: ' + (active['memory'] if active else 'не выбрана'))
        print('1 Открыть память\n2 Создать резервную копию\n3 Восстановить копию\n4 Выбрать рабочую память\n0 Выход')
        try:
            choice = input('Действие: ').strip()
            if choice == '0':
                return
            if choice == '1':
                manager.launch()
            elif choice == '2':
                print('Приложение должно быть закрыто. Копия без шифрования. Лимит: ' + str(budget // 1024**2) + ' МиБ.')
                record = manager.create_backup(input('Новая папка копии (полный путь): ').strip(), budget)
                print('Копия проверена. Сохраните отдельно папку и контрольную сумму:\n' + json.dumps(record, ensure_ascii=False))
            elif choice == '3':
                records = [read_json(p) for p in sorted((manager.state / 'backups').glob('*.json'))]
                manual = input('1 Выбрать сохранённую копию; 2 Указать копию и сохранённый SHA-256: ').strip()
                record = choose(records, lambda r: r['backup']) if manual == '1' else (
                    dict(backup=input('Папка копии: ').strip(), sha256=input('Ранее сохранённый SHA-256: ').strip()) if manual == '2' else None)
                if record is None:
                    continue
                decoder = active['precompressor'] if active else None
                if decoder is None:
                    decoder = input('Путь к доверенному Precomp (Enter, если не нужен): ').strip() or None
                ident = manager.restore_backup(record['backup'], record['sha256'],
                                               input('Новая папка восстановления: ').strip(),
                                               Path(decoder) if decoder else None, budget)
                print('Восстановление проверено. Рабочая память не переключена. Выберите её пунктом 4. Настройка: ' + ident)
            elif choice == '4':
                ids = [p.stem for p in sorted((manager.state / 'profiles').glob('*.json'))]
                ident = choose(ids, lambda i: manager.get(i)['memory'])
                if ident and input('Использовать эту папку? Напишите ДА: ').strip() == 'ДА':
                    manager.activate(ident)
            else:
                print('Выберите действие от 0 до 4.')
        except (OSError, ValueError, RuntimeError, KeyError, ArchiveError) as error:
            print('Действие не завершено: ' + str(error))
            print('Если память занята — закройте её приложение. Частично созданную папку не используйте; укажите новую.')


def main():
    parser = argparse.ArgumentParser(description='GLYPH: запуск, резервная копия и восстановление')
    parser.add_argument('--home', type=Path, default=Path(os.environ.get('GLYPH_PILOT_HOME', str(Path.home() / 'GlyphPilot'))))
    parser.add_argument('--state', type=Path)
    parser.add_argument('--max-bytes', type=int, default=1024**3)
    args = parser.parse_args()
    if not 0 < args.max_bytes <= 64 * 1024**3:
        parser.error('лимит должен быть от 1 байта до 64 ГиБ')
    manager = Manager(args.state or args.home / 'personal-manager')
    try:
        with manager.lock():
            if not (manager.state / 'ACTIVE').exists():
                try:
                    ident = manager.save_profile(discover(args.home))
                    manager.activate(ident)
                except (OSError, ValueError, ArchiveError) as error:
                    print('Автонастройка недоступна: ' + str(error) + '\nМожно восстановить копию пунктом 3.')
            menu(manager, args.max_bytes)
    except BlockingIOError:
        raise SystemExit('Помощник уже открыт. Вернитесь в его окно.')
    except (EOFError, KeyboardInterrupt):
        print('\nПомощник закрыт.')


if __name__ == '__main__':
    main()
