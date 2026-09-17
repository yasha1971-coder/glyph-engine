#!/usr/bin/env python3
"""Reproducible synthetic growth experiment; no user data read."""
import json
import random
import sys
import time
import zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))
from test_incremental_memory import IncrementalMemoryTests
import incremental_memory as inc
import chunk_versions as cv


def main():
    f = IncrementalMemoryTests()
    f.setUp()
    try:
        m = f.m
        files = m.snapshot(f.first)['files']
        for i in range(414):
            files['placeholder-%03d.txt' % i] = dict(files['note.txt'])
        pin = m.commit(f.first, files)
        data = random.Random(1984).randbytes(512 * 1024)
        entries, whole_bytes = [], 0
        start = time.monotonic()
        for revision in range(101):
            if revision:
                # Prefix insertion defeats fixed-alignment dedup; one byte edit
                # probes local change handling at a second location.
                mutable = bytearray(data)
                offset = (revision * 3571) % len(mutable)
                mutable[offset] ^= 1
                data = bytes([revision % 256]) + mutable
            (f.src / 'versioned.bin').write_bytes(data)
            pin = m.add(pin, f.src)
            entries.append((pin, inc.digest(data), len(data)))
            whole_bytes += min(len(data), len(zlib.compress(data, 6)))
        for pin, sha, size in entries:
            output = m.read_item(m.snapshot(pin)['files']['versioned.bin'])
            assert len(output) == size and inc.digest(output) == sha
        sizes = {}
        for name in ('objects', 'chunks', 'recipes', 'snapshots'):
            root = m.root / name
            sizes[name] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
        result = {'format': 'GLYPH_SYNTHETIC_VERSION_GROWTH_V1',
                  'chunk_min': cv.MIN, 'chunk_target': cv.TARGET, 'chunk_max': cv.MAX,
                  'seed': 1984, 'initial_file_bytes': 512 * 1024, 'edits': 100,
                  'versions_verified': len(entries), 'catalog_paths_after_add': 416,
                  'whole_file_payload_baseline_bytes': whole_bytes,
                  'actual_overlay_bytes_by_directory': sizes,
                  'actual_overlay_logical_bytes': sum(sizes.values()),
                  'note': 'Baseline excludes metadata; actual includes all snapshots, recipes and payloads. Synthetic insertion+byte edits, not VIKA or a universal ratio. Base archive excluded from both.',
                  'elapsed_seconds': round(time.monotonic() - start, 3)}
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        f.doCleanups()


if __name__ == '__main__':
    main()
