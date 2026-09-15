import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import golden_demo as demo
import qualification as q
import hashlib

class QualificationTests(unittest.TestCase):
    def test_quantile_and_empty(self):
        self.assertEqual(q.summary([5,1,3,2,4])['median'],3)
        self.assertEqual(q.summary([5,1,3,2,4])['p95'],5)
        with self.assertRaises(ValueError): q.summary([])

    def test_fresh_workers_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir()
            data=b'<xml>synthetic</xml>\n'*100
            (source/'xml').write_bytes(data)
            with patch.object(demo,'SILESIA',{'xml':(len(data),hashlib.md5(data).hexdigest())}), contextlib.redirect_stdout(io.StringIO()):
                demo.build(source,root/'demo',revisions=3)
            for kind,n in [('base',1),('history',4),('deflate6',1)]:
                row=q.launch(root/'demo',kind)
                self.assertEqual(len(row['samples']),n)
                self.assertGreater(row['peak_rss_bytes'],0)
            p=root/'demo'/demo.REPORT
            p.write_bytes(p.read_bytes()+b' ')
            with self.assertRaises(ValueError): q.receipt(root/'demo')
