import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import recovery_recipes as rr
from chunk_versions import pack


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name);self.data=self.root/'data';self.tx='a'*32
        d=self.data/'packs'/self.tx;d.mkdir(parents=True)
        self.identity={'format':'GLYPH_PACKED_SCALE_STORE_V1','uuid':'b'*32}
        (self.data/'identity.json').write_text(json.dumps(self.identity))
        self.body='Исходный текст. оригінал'.encode(); sha=hashlib.sha256(self.body).digest();p=pack(self.body)
        self.path=d/'00000000.pack';self.path.write_bytes(rr.HEADER.pack(b'GLYPHPK1',len(self.body),len(p),sha)+p)
        b=dict(tx=self.tx,pack=0,offset=0,bytes=len(self.body),stored=len(p),sha256=sha.hex())
        self.doc=dict(format=rr.FORMAT,identity=self.identity,versions=[dict(id=1,name='note',created_ns=123,bytes=len(self.body),sha256=sha.hex(),blocks=[b])])

    def encoded(self):
        raw=json.dumps(self.doc).encode();return raw,hashlib.sha256(raw).hexdigest()

    def test_no_database_required(self):
        raw,pin=self.encoded()
        self.assertEqual(rr.restore(raw,pin,self.data,1),self.body)
        self.assertEqual(rr.load(raw,pin)['versions'][0]['created_ns'],123)

    def test_pin_corruption_missing_pack_wrong_identity(self):
        raw,pin=self.encoded()
        with self.assertRaises(ValueError):rr.restore(raw+b' ',pin,self.data,1)
        self.path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):rr.restore(raw,pin,self.data,1)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):rr.restore(raw,pin,self.data,1)
        (self.data/'identity.json').write_text('{}')
        with self.assertRaises(ValueError):rr.restore(raw,pin,self.data,1)

    def test_traversal_duplicate_and_unknown_version(self):
        raw,pin=self.encoded()
        with self.assertRaises(ValueError):rr.restore(raw,pin,self.data,2)
        self.doc['versions'][0]['blocks'][0]['tx']='../escape'
        raw,pin=self.encoded()
        with self.assertRaises(ValueError):rr.restore(raw,pin,self.data,1)
        self.doc['versions']*=2
        raw,pin=self.encoded()
        with self.assertRaises(ValueError):rr.load(raw,pin)

    @unittest.skipUnless(importlib.util.find_spec('packed_scale_store'),'experimental backend unavailable')
    def test_real_store_catalogue_loss_preserves_versions(self):
        from packed_scale_store import Store
        m,d=self.root/'meta',self.root/'real-data'
        with Store(m,d,create=True) as s:
            a=s.put_version('document',[b'original '*1000])
            b=s.put_version('document',[b'original '*1000,b'edit'])
            raw,pin=rr.export(s)
        m.rename(self.root/'unavailable-catalogue')
        self.assertFalse(m.exists())
        self.assertEqual(rr.restore(raw,pin,d,a),b'original '*1000)
        self.assertEqual(rr.restore(raw,pin,d,b),b'original '*1000+b'edit')
        self.assertEqual(len(rr.load(raw,pin)['versions']),2)
