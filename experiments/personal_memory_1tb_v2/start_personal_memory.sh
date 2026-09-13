#!/usr/bin/env bash
# Reuse the completed, pinned archive; never scan or recompress source files.
set -euo pipefail
BASE=${GLYPH_PILOT_HOME:-"$HOME/GlyphPilot"}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 - "$BASE" "$HERE" <<'PY'
import hashlib
from pathlib import Path
import os
import sys
base, here = map(Path, sys.argv[1:])
pin = 'e3253b6e1ef269a85f94ec632d581ffffe8df22ed554b2b46e9c86f8adeddf55'
receipts = list((base / 'GLYPH-V2-runs').glob('*/GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json'))
matches = [p.parent for p in receipts if not p.is_symlink() and p.stat().st_size <= 8*1024*1024
           and hashlib.sha256(p.read_bytes()).hexdigest() == pin]
if len(matches) != 1:
    raise SystemExit('STOP: требуется ровно один готовый архив с ожидаемым хешем; найдено: ' + str(len(matches)))
args = [sys.executable, str(here / 'memory_browser.py'),
        '--memory', str(base / 'GLYPH-V2-runs/personal-memory-ui-v1'),
        '--archive', str(matches[0]), '--archive-sha256', pin,
        '--precompressor', str(base / 'tools/precomp-v0.4.7/linux/precomp'),
        '--precompressor-sha256', '6a5289ba81ea658e8e7984bf32791635f53b8e7cee4aafc26b853c7cd5b018f8']
os.execv(sys.executable, args)
PY
