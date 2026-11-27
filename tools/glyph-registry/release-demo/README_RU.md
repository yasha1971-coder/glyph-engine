# GLYPH №3 — локальный пользовательский кандидат

Готовые программы Linux x86-64, публичная выборка Git v2.49.0:2255 файлов,
коммит683c54c999c301c2cd6f715c411407c413b1d84e. Основной формат — релизный
RLB1/RLR1/LOC2; native reader и №1 из glyph-01-02-v0.1.0-rc1 не изменены.
Нужны Ubuntu24.04/WSL, Python3.12stdlib и установленный bubblewrap (`bwrap`).
Компиляция, сеть, исходный checkout и отдельный plaintext corpus не нужны.

Сначала получите SHA архива от Якова по доверенному каналу, независимо от автора
результата. Сверьте с ним `.sha256`, затем в новой папке WSL:

```bash
sha256sum -c glyph03-candidate-linux-x86_64.tar.gz.sha256
tar -xzf glyph03-candidate-linux-x86_64.tar.gz
cd glyph03-candidate-linux-x86_64
sha256sum -c CHECKSUMS.sha256
```

Две проверяемые команды; каждая work-папка должна отсутствовать:

```bash
python3 demo.py standalone --query ALLOC_GROW --work /tmp/glyph03-alone-1
python3 demo.py chain --query-file acceptance/chain.query.bin --work /tmp/glyph03-chain-1
```

№3 отдельно: `ok=true`, count252, returned10, byte_check=true, files_verified2255.
В stdout и `/tmp/glyph03-alone-1/summary.json` будут10 разных пар `(path, offset)`;
offset — нулевое байтовое смещение внутри файла. Допустимы любые10 настоящих
совпадений, без обещания первых10 по тексту. Введите свой точный запрос через
`--query 'your literal'` или `--query-file /path/query.bin`. UTF-8,1..1024байта,
без служебных00/01; переводы строк/кодировка не меняются.

Связка №3 → №2 → №1:
1. Издательская копия verifier+anchor передаётся B ДО выбора запроса A.
2. A получает индекс, затем читает выбранный новый запрос. Индекс не перестраивается;
   SHA каждой его части до/после запроса проверяется. №2 находит новый результат,
   реальный `glyph-evidence create-index` создаёт evidence.
3. A передаёт B `second-receiver/incoming/result-data.tar.gz`: только9 файлов данных,
   включая индекс, запрос и result. Код/`RUN.sh` автора результата не исполняется.
4. B запускает свой `verify_handoff.py` и неизменный штатный №1. Он сверяет исходный
   registry/core/corpus PIN, полный overlapping count и байты, повторяет native query,
   восстанавливает дерево из BWT и перечитывает SHA каждого записанного файла.
5. Отдельная свежая папка B проверяет подмену count1→2: связи result и транспортный
   SHA пересчитаны, но проверка обязана отказать code1 с `count or coverage mismatch`.

Ожидаемый успех: count1, returned1, path `t/t1003-read-tree-prefix.sh`, offset234,
files_verified2255, byte_check/native_replay=true. Ожидаемый отказ сохранён отдельно:
`B-expected-rejection.command.json` содержит exit_code1, `.stderr` — причину.
Общий demo возвращает0 только если успех получен И ожидаемый отказ подтверждён.
Все реальные команды/stdout/stderr/коды — в work-папке, итог — `summary.json`;
дерево B — `second-receiver/positive/state/restored/`, порядок этапов — `stages.json`.
Для своего нового запроса замените путь `--query-file`; count изменится соответственно.

Источник доверия B — этот издательский архив с независимо полученным SHA,
а не поля, контрольные суммы или программы из результата A. Скопируйте из него
`verify_handoff.py`, `TRUSTED_CODE.json`, `runtime/`, `licenses/`, `trust/` в
отдельную trusted-tools папку до получения результата. Зафиксируйте:
- corpus SHA: `a2959331768bffeaeeadf9bda22a5e64ee24158ebc347a33ed697ae36481e5a3`;
- registry/index SHA: `3c937033273babffd9a8445a6fd40000d4d2a2c6e7373f17d94110d8cb4bf607`;
- core manifest SHA: `7c8a60d8e380787beeef022581937c0dffebb9a72f606b79c42543862bd667b6`;
- documents SHA: `89605b49c1ded32cb0eb04d1eebabc07fd5a06aaa460c4d9bcb7d03ad6b3d8c2`;
- SHA файла `trust/Git-anchor.json`: `655918b17755a5d9cb9af1872fbf3509429500eb82b0cc5ba46cd2de5860cf6c`.
Хеш выбранного запроса и limit10 поступают B из согласованного задания отдельно
от проверяемого заявления. Развёрнутая команда B записана в `B-verify-success.command.json`.
Капсула — данные; B не принимает из неё новые trust anchors или программы.

Граница: профиль с закреплённым индексом, `canonical_runtime_rebuilt=false`;
это не canonical-перестроение. Архив уже на диске; сеть не входит в проверку.
Ожидания зафиксированы до приёмки: `acceptance/EXPECTED.json`. Отдельный контролёр
запускает обе команды из окончательного архива и сверяет их с прямым поиском по
исходным публичным байтам вне source-free sandbox. Его приватные логи не в архиве.
Решение RLB2 и правило стоимости преобразований — `DECISION_RLB2.md`.
Всё локально. Публикацию решает Яков.
