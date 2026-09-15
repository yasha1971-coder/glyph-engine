from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scale_lab as lab

class ScaleLabTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.source=self.root/'source';self.source.mkdir()
        (self.source/'sample.txt').write_bytes(b'public synthetic data\n'*100)
        (self.source/'тут все.txt').write_bytes(b'EXCLUDED TEST MARKER')
        self.out=self.root/'lab'
        self.p=patch.object(lab,'BLOCK',128*1024);self.p.start();self.addCleanup(self.p.stop)
        self.c=lab.prepare(self.source,self.out,8*lab.BLOCK+19,0)

    def test_resume_roundtrip_and_exclusion(self):
        self.assertNotIn(lab.sha(b'EXCLUDED TEST MARKER'),self.c['seeds'])
        self.assertFalse(lab.run(self.out,self.c,max_blocks=3))
        self.assertFalse((self.out/'RESULT.json').exists())
        self.assertTrue(lab.run(self.out,self.c))
        r=lab.verify(self.out,self.c)
        self.assertEqual(r['logical_input_bytes'],self.c['target_bytes'])
        self.assertEqual(r['blocks'],9)
        self.assertTrue((self.source/'тут все.txt').exists())

    def test_corrupt_object_stops_resume(self):
        lab.run(self.out,self.c,max_blocks=1)
        obj=next((self.out/'objects').iterdir());obj.write_bytes(b'bad')
        with self.assertRaises(ValueError):lab.run(self.out,self.c)
        self.assertFalse((self.out/'RESULT.json').exists())

    def test_orphan_after_interrupt_and_no_overwrite(self):
        data,kind=lab.source_block(self.c,self.out,0)
        lab.atomic(self.out/'objects'/lab.sha(data),data)
        self.assertTrue(lab.run(self.out,self.c))
        lab.verify(self.out,self.c)
        with self.assertRaises(ValueError):lab.prepare(self.source,self.out,1,0)

    def test_disk_reserve_stops_without_progress(self):
        self.c['reserve_bytes']=10**30
        with self.assertRaises(RuntimeError):lab.run(self.out,self.c)
        db=lab.database(self.out)
        try:self.assertEqual(db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0],0)
        finally:db.close()
