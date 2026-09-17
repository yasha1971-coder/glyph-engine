import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chunk_versions as c
import native_cdc as n

@unittest.skipUnless(shutil.which('g++'),'native tests require g++')
class NativeCDCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.lib=str(Path(cls.tmp.name)/'gear.so')
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(Path(c.__file__).with_name('gear_native.cpp')),'-o',cls.lib],check=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_identical_boundaries_and_content(self):
        rng=random.Random(710)
        sizes=[0,1,4095,4096,4097,16383,16384,65535,65536,65537,1024*1024]
        cases=[b'\0'*s for s in sizes]+[b'\xff'*s for s in sizes]+[rng.randbytes(s) for s in sizes]
        cases += [rng.randbytes(rng.randrange(300000)) for _ in range(30)]
        for data in cases:
            expected=list(c.split_python(data))
            self.assertEqual(list(n.split(data,c.GEAR,c.MIN,c.TARGET,c.MAX,self.lib)),expected)
        with patch.dict(os.environ,{'GLYPH_CDC_NATIVE':self.lib}):
            self.assertEqual(list(c.split(cases[-1])),list(c.split_python(cases[-1])))
    def test_other_parameters_and_insertions(self):
        b=random.Random(72).randbytes(300000)
        for minimum,target,maximum in [(1024,8192,32768),(16384,32768,131072)]:
            with patch.multiple(c,MIN=minimum,TARGET=target,MAX=maximum):
                for data in [b,b'x'+b,b[71:],b[:99]+b[199:]]:
                    self.assertEqual(list(n.split(data,c.GEAR,minimum,target,maximum,self.lib)),list(c.split_python(data)))
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):list(n.split(b'x',c.GEAR,0,16,64,self.lib))
