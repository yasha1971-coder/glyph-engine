import json
from pathlib import Path
import random
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import codec_frontier as f

class CodecFrontierTests(unittest.TestCase):
    def test_all_policies_roundtrip_and_raw_fallback(self):
        try:z=f.Zstd()
        except (OSError,RuntimeError):z=None
        policies=list(f.POLICIES)+(['zstd-3','zstd-9'] if z else [])
        for b in [b'',random.Random(13).randbytes(4096),b'valid public record 1\n'*12000]:
            for policy in policies:
                codec,payload,_=f.select(policy,b,z)
                self.assertLessEqual(len(payload),len(b))
                self.assertEqual(f.decode(codec,payload,len(b),z),b)
    def test_corrupt_zstd_rejected(self):
        try:z=f.Zstd()
        except (OSError,RuntimeError):self.skipTest('libzstd unavailable')
        with self.assertRaises(ValueError):z.decompress(b'bad frame',10)
    def test_new_output_workers_source_identity_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'source';s.mkdir();(s/'ok.txt').write_bytes(b'public fixture'*300)
            (s/'тут все.txt').write_bytes(b'NEVER INCLUDE TEST MARKER')
            report=f.run(s,r/'out',1,2,policies=['bzip2-9','sample-bz-xz6'])
            self.assertEqual(report['source_files'],1)
            self.assertTrue(all(x['exact_restore'] for v in report['runs'] for x in v['files']))
            with self.assertRaises(ValueError):f.run(s,r/'out',1,2)
            with self.assertRaises(ValueError):f.job((str(s/'ok.txt'),'0'*64,'bzip2-9',str(r/'bad.tmp')))
            self.assertFalse((r/'bad.tmp').exists())
            self.assertTrue((s/'тут все.txt').is_file())
    def test_sample_is_bounded_and_no_prior_results(self):
        b=random.Random(73).randbytes(1024*1024)
        choice,info=f.probe(b)
        self.assertEqual(info['sample_bytes'],192*1024)
        self.assertEqual(f.probe(b),(choice,info))
        self.assertEqual(f.probe(b'hello')[1]['sample_bytes'],0)
