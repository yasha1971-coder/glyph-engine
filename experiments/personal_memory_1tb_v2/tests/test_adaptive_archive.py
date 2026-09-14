import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adaptive_codec as adaptive
import codec_frontier as frontier
import incremental_memory as inc
import permanent_delete as deletion
import verified_hybrid_archive as archive
from test_verified_hybrid_archive import make_inventory


class AdaptiveArchiveTests(unittest.TestCase):
    def test_frozen_rule_and_payload_match_benchmark(self):
        for data in (b'', b'hello', random.Random(7).randbytes(220000),
                     b'public record\n' * 60000):
            expected = frontier.select(adaptive.POLICY, data)
            self.assertEqual(adaptive.encode(data), expected[:2])

    def test_parallel_dedup_restore_and_byte_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            expected = {'a': b'public data\n' * 30000, 'b': b'public data\n' * 30000,
                        'random': random.Random(2).randbytes(20000), 'empty': b''}
            for name, data in expected.items(): (source / name).write_bytes(data)
            state = root / 'inventory'; make_inventory(source, state)
            with patch.object(archive, '_encode_verified', wraps=archive._encode_verified) as encoded:
                m = archive.build(state, root / 'archive', 0, codec_policy=adaptive.POLICY, workers=2)
                self.assertEqual(encoded.call_count, 3)
            self.assertEqual(m['unique_objects'], 3)
            self.assertEqual(m['duplicate_file_references'], 1)
            self.assertEqual(sum(p.stat().st_size for p in (root / 'archive').rglob('*') if p.is_file()),
                             m['container_logical_bytes'])
            archive.restore(root / 'archive', root / 'restored')
            self.assertEqual({p.name: p.read_bytes() for p in (root / 'restored').iterdir()}, expected)

    def test_xz6_cold_memory_versions_delete_and_corrupt_object(self):
        # Force the xz branch for decoder coverage, not to score policy quality.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            data = random.Random(54).randbytes(50000) * 8
            (source / 'note.bin').write_bytes(data)
            state = root / 'inventory'; make_inventory(source, state)
            with patch.object(adaptive, 'choose', return_value='xz-6'):
                report = archive.build(state, root / 'archive', 0, codec_policy=adaptive.POLICY, workers=2)
            self.assertEqual(report['codec_object_counts'], {'xz-6': 1})
            self.assertEqual(sum(p.stat().st_size for p in (root / 'archive').rglob('*') if p.is_file()),
                             report['container_logical_bytes'])
            pin = inc.digest((root / 'archive' / archive.RECEIPT).read_bytes())
            m = inc.Memory(root / 'memory', root / 'archive', pin)
            first = m.initialize(); deletion.set_head(m, first)
            self.assertEqual(m.read_item(m.snapshot(first)['files']['note.bin']), data)
            incoming = root / 'incoming'; incoming.mkdir()
            (incoming / 'note.bin').write_bytes(b'prefix' + data)
            second = m.add(first, incoming); deletion.set_head(m, second)
            cold = inc.Memory(root / 'memory', root / 'archive', pin)
            self.assertEqual(cold.read_item(cold.snapshot(second)['files']['note.bin']), b'prefix' + data)
            current = deletion.delete(cold, second, 'note.bin', first)
            self.assertEqual(cold.read_item(cold.snapshot(current)['files']['note.bin']), b'prefix' + data)
            obj = root / 'archive' / report['objects'][0]['path']
            obj.write_bytes(obj.read_bytes() + b'corrupt')
            with self.assertRaises(archive.ArchiveError):
                m.backend.read_verified('note.bin', len(data), inc.digest(data))
            self.assertFalse((root / 'bad-restore').exists())

    def test_worker_failure_and_invalid_options_publish_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            (source / 'file').write_bytes(b'restore me')
            state = root / 'inventory'; make_inventory(source, state)
            with patch.object(adaptive, 'encode', return_value=('raw', b'wrong')):
                with self.assertRaises(archive.ArchiveError):
                    archive.build(state, root / 'bad', 0, codec_policy=adaptive.POLICY, workers=2)
            self.assertFalse((root / 'bad').exists())
            self.assertFalse(list(root.glob('.bad.tmp-*')))
            for options in ({'workers': 3}, {'codec_policy': 'unknown'}, {'workers': True}):
                with self.assertRaises(archive.ArchiveError): archive.build(state, root / 'bad', 0, **options)
            (source / 'тут все.txt').write_bytes(b'EXCLUSION TEST MARKER')
            state2 = root / 'inventory2'; make_inventory(source, state2)
            with self.assertRaises(archive.ArchiveError):
                archive.build(state2, root / 'bad', 0, codec_policy=adaptive.POLICY)
            self.assertTrue((source / 'тут все.txt').is_file())


if __name__ == '__main__':
    unittest.main()
