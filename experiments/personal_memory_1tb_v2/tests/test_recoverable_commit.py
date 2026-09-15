import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import recoverable_commit as commit
import recovery_recipes as recipes


@unittest.skipUnless(importlib.util.find_spec('packed_scale_store'),'experimental backend unavailable')
class CommitTests(unittest.TestCase):
    def test_kill_boundaries_preserve_acknowledged_version_and_reconcile(self):
        from packed_scale_store import Store
        code='''
import os,signal,sys
from packed_scale_store import Store
from recoverable_commit import commit_version
with Store(sys.argv[1],sys.argv[2]) as s:
 commit_version(s,sys.argv[3],'document',[b'new'],fault=lambda stage: os.kill(os.getpid(),signal.SIGKILL) if stage==sys.argv[4] else None)
'''
        for boundary in ('after_store_commit','after_capsule_fsync','after_capsule_publish','after_head_publish'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as t:
                root=Path(t);m,d,r=root/'meta',root/'data',root/'recovery'
                with Store(m,d,create=True) as s:
                    first=commit.commit_version(s,r,'document',[b'old'])
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
                p=subprocess.run([sys.executable,'-c',code,str(m),str(d),str(r),boundary],env=env,timeout=20)
                self.assertEqual(p.returncode,-signal.SIGKILL)
                raw,pin=commit.current(r)
                self.assertEqual(recipes.restore(raw,pin,d,first['version']),b'old')
                self.assertEqual(len(recipes.load(raw,pin)['versions']),2 if boundary=='after_head_publish' else 1)
                with Store(m,d) as s:
                    new_pin=commit.publish(s,r)
                    self.assertEqual(len(s.history('document')),2)
                raw,pin=commit.current(r,expected_pin=new_pin)
                self.assertEqual(recipes.restore(raw,pin,d,2),b'new')

    def test_success_returns_recoverable_pin_and_wrong_store_rejected(self):
        from packed_scale_store import Store
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);r=root/'recovery'
            with Store(root/'m',root/'d',create=True) as s:
                result=commit.commit_version(s,r,'document',[b'one'])
                raw,pin=commit.current(r,expected_pin=result['recovery_pin'])
                self.assertEqual(recipes.restore(raw,pin,s.data,result['version']),b'one')
            with Store(root/'m2',root/'d2',create=True) as s:
                with self.assertRaises(ValueError):commit.commit_version(s,r,'other',[b'two'])
                self.assertEqual(s.stats()['versions'],0)
