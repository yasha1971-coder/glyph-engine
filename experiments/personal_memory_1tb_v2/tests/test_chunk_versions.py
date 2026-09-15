import json
import random
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import incremental_memory as inc
import chunk_versions as cv
import test_incremental_memory as fixture


class ChunkVersionTests(unittest.TestCase):
    def setUp(self):
        f = fixture.IncrementalMemoryTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.m, self.src, self.pin = f.m, f.src, f.first
        self.data = random.Random(847).randbytes(1024 * 1024)

    def add(self, data):
        (self.src / 'large.bin').write_bytes(data)
        self.pin = self.m.add(self.pin, self.src)
        return self.m.snapshot(self.pin)['files']['large.bin']

    def test_insertion_reuses_ranges_and_no_delta_chain(self):
        old = self.add(self.data)
        oldpin = self.pin
        data = b'inserted prefix' + self.data
        item = self.add(data)
        self.assertEqual(item['storage'], 'chunks-v1')
        self.assertEqual(self.m.read_item(item), data)
        doc = cv.recipe(self.m, item)
        self.assertGreater(sum(c['bytes'] for c in doc['chunks'] if 'source' in c), len(data) * .7)
        third = self.add(data + b'tail')
        self.assertEqual(self.m.read_item(third), data + b'tail')
        for c in cv.recipe(self.m, third)['chunks']:
            if 'source' in c:
                self.assertNotEqual(c['source']['storage'], 'chunks-v1')
        self.assertEqual(self.m.read_item(self.m.snapshot(oldpin)['files']['large.bin']), self.data)

    def test_compressible_data_uses_whole_fallback(self):
        item = self.add(b'A' * 100000)
        self.assertEqual(item['storage'], 'deflate')

    def test_missing_recipe_corrupt_chunk_and_range_bounds(self):
        self.add(self.data)
        item = self.add(b'prefix' + self.data)
        doc = cv.recipe(self.m, item)
        plain = next(c for c in doc['chunks'] if 'source' not in c)
        path = self.m.root / 'chunks' / plain['sha256']
        payload = path.read_bytes()
        path.write_bytes(b'Rbad')
        with self.assertRaises(Exception):
            self.m.read_item(item)
        path.write_bytes(payload)
        ranged = next(c for c in doc['chunks'] if 'source' in c)
        ranged['offset'] = -1
        raw = inc.base.canonical_json(doc)
        h = inc.digest(raw)
        (self.m.root / 'recipes' / h).write_bytes(raw)
        bad = dict(item, recipe=h)
        with self.assertRaises(inc.Error):
            self.m.read_item(bad)
        (self.m.root / 'recipes' / item['recipe']).unlink()
        with self.assertRaises(Exception):
            self.m.read_item(item)

    def test_legacy_snapshot_stays_readable(self):
        doc = self.m.snapshot(self.pin)
        doc['format'] = inc.LEGACY_FORMAT
        doc.pop('created_ns', None)
        raw = inc.base.canonical_json(doc)
        h = inc.digest(raw)
        (self.m.root / 'snapshots' / (h + '.json')).write_bytes(raw)
        self.assertEqual(self.m.snapshot(h)['files'], doc['files'])

    def test_chunking_is_deterministic_and_bounded(self):
        pieces = list(cv.split(self.data))
        self.assertEqual(pieces, list(cv.split(self.data)))
        self.assertEqual(b''.join(x[1] for x in pieces), self.data)
        self.assertTrue(all(cv.MIN <= len(p) <= cv.MAX for _, p in pieces[:-1]))

    def test_packed_snapshot_and_invalid_stream(self):
        import base64
        import zlib
        files = self.m.snapshot(self.pin)['files']
        for i in range(100):
            files[str(i)] = dict(files['note.txt'])
        pin = self.m.commit(self.pin, files)
        raw = (self.m.root / 'snapshots' / (pin + '.json')).read_bytes()
        self.assertEqual(json.loads(raw)['format'], 'GLYPH_SNAPSHOT_PACK_V1')
        self.assertEqual(self.m.snapshot(pin)['files'], files)
        envelope = {'format': 'GLYPH_SNAPSHOT_PACK_V1', 'deflate_base64': base64.b64encode(zlib.compress(b'{}') + b'junk').decode()}
        bad = inc.base.canonical_json(envelope)
        h = inc.digest(bad)
        (self.m.root / 'snapshots' / (h + '.json')).write_bytes(bad)
        with self.assertRaises(inc.Error):
            self.m.snapshot(h)
