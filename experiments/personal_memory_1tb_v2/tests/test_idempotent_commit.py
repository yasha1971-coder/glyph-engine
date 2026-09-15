import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from idempotent_commit import commit_once
import recoverable_commit as recovery
import recovery_recipes as recipes


@unittest.skipUnless(importlib.util.find_spec('packed_scale_store'),'experimental backend unavailable')
class RetryTests(unittest.TestCase):
    def test_replay_conflict_and_distinct_operations(self):
        from packed_scale_store import Store
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)
            with Store(r/'m',r/'d',create=True) as s:
                first=commit_once(s,r/'recovery','op1','note',[b'original'])
                second=commit_once(s,r/'recovery','op1','note',[b'original'])
                self.assertEqual(first['version'],second['version']);self.assertTrue(second['replayed'])
                for name,blocks in [('note',[b'changed']),('another',[b'original']),('note',[b'orig',b'inal'])]:
                    with self.assertRaises(ValueError):commit_once(s,r/'recovery','op1',name,blocks)
                self.assertEqual(s.stats()['versions'],1)
                third=commit_once(s,r/'recovery','op2','note',[b'original'])
                self.assertNotEqual(first['version'],third['version'])
                self.assertEqual(s.stats()['objects'],1)

    def test_real_sigkill_and_retry_create_exactly_one_version(self):
        from packed_scale_store import Store
        child='''
import os,signal,sys
from packed_scale_store import Store
from idempotent_commit import commit_once
def fault(stage):
 if stage==sys.argv[4]:os.kill(os.getpid(),signal.SIGKILL)
with Store(sys.argv[1],sys.argv[2],fault=fault) as s:
 commit_once(s,sys.argv[3],'request-1','note',[b'new data'],fault=fault)
'''
        for boundary in ('after_intent','after_pack_publish','after_commit','after_store_commit','after_capsule_publish','after_head_publish'):
            with self.subTest(boundary=boundary),tempfile.TemporaryDirectory() as t:
                r=Path(t);m,d,h=r/'m',r/'d',r/'recovery'
                with Store(m,d,create=True) as s:recovery.publish(s,h)
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
                p=subprocess.run([sys.executable,'-c',child,str(m),str(d),str(h),boundary],env=env,timeout=20)
                self.assertEqual(p.returncode,-signal.SIGKILL)
                with Store(m,d) as s:
                    result=commit_once(s,h,'request-1','note',[b'new data'])
                    self.assertEqual(s.stats()['versions'],1)
                    self.assertEqual(s.db.execute('SELECT count(*) FROM retry_operations_v1 WHERE version IS NOT NULL').fetchone()[0],1)
                    self.assertEqual(commit_once(s,h,'request-1','note',[b'new data'])['version'],result['version'])
                raw,pin=recovery.current(h,expected_pin=result['recovery_pin'])
                self.assertEqual(recipes.restore(raw,pin,d,result['version']),b'new data')
