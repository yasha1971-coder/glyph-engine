import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import golden_demo as demo
import incremental_memory as inc


class GoldenDemoTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.golden = self.root / 'GOLDEN'; self.golden.mkdir()
        self.data = b'<?xml version="1.0"?><text>synthetic only</text>\n' * 100
        (self.golden / 'xml').write_bytes(self.data)
        self.profile = {'xml': (len(self.data), hashlib.md5(self.data).hexdigest())}

    def test_synthetic_workflow_and_original_unchanged(self):
        with patch.object(demo, 'SILESIA', self.profile), contextlib.redirect_stdout(io.StringIO()):
            report = demo.build(self.golden, self.root / 'demo', revisions=3)
        self.assertEqual(report['status'], 'LOCAL_GATES_PASSED')
        self.assertEqual(len(report['versions']), 4)
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual((self.golden / 'xml').read_bytes(), self.data)
        raw = (self.root / 'demo' / demo.REPORT).read_bytes()
        self.assertEqual((self.root / 'demo' / (demo.REPORT + '.sha256')).read_text().strip(), inc.digest(raw))
        self.assertEqual(len(report['source_files']), 1)

    def test_missing_bad_hash_existing_output_no_writes(self):
        with patch.object(demo, 'SILESIA', {'xml': (len(self.data), '0' * 32)}):
            with self.assertRaises(inc.Error):
                demo.build(self.golden, self.root / 'demo')
        self.assertFalse((self.root / 'demo').exists())
        (self.root / 'demo').mkdir()
        with self.assertRaises(inc.Error):
            demo.build(self.golden, self.root / 'demo')
        self.assertEqual(list((self.root / 'demo').iterdir()), [])

    def test_source_output_overlap_is_rejected(self):
        with self.assertRaises(inc.Error):
            demo.build(self.golden, self.golden / 'output')

    def test_official_profile_size_and_count(self):
        self.assertEqual(len(demo.SILESIA), 12)
        self.assertEqual(sum(x[0] for x in demo.SILESIA.values()), 211938580)

    def test_staged_aliases_and_adaptive_full_workflow(self):
        (self.golden / 'xml').rename(self.golden / 'xml.xml')
        with patch.object(demo, 'SILESIA', self.profile), contextlib.redirect_stdout(io.StringIO()):
            report = demo.build(self.golden, self.root / 'demo', revisions=3,
                                codec_policy='sample-bz-xz6', workers=2)
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(report['base_codec_policy'], 'sample-bz-xz6')
        self.assertEqual(report['base_codec_workers'], 2)
        self.assertEqual((self.golden / 'xml.xml').read_bytes(), self.data)
