import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediator import execute, local_plan, parse

class SyntheticView:
    version = '1' * 64
    def __init__(self, corrupt=False):
        self.content = {'note.txt': 'Демонстрация: встреча 18 сентября.'.encode(),
                        'other.txt': b'private fixture, never granted'}
        self.files = {p: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for p,b in self.content.items()}
        self.corrupt = corrupt
        self.reads = []
    def read(self, path):
        self.reads.append(path)
        if self.corrupt: raise ValueError('synthetic integrity failure')
        return self.content[path]

class MediatorTests(unittest.TestCase):
    def test_exact_query_returns_bound_evidence(self):
        v=SyntheticView()
        r=execute(json.dumps(dict(action='search',query='18 сентября',question='')), [v], {(v.version,'note.txt')})
        self.assertEqual(r['status'],'FOUND')
        e=r['evidence'][0]
        self.assertEqual(e['sha256'],v.files['note.txt']['sha256'])
        self.assertEqual(e['text'].encode(),v.content['note.txt'][e['byte_offset']:e['byte_offset']+e['byte_length']])
        self.assertEqual(v.reads,['note.txt'])
    def test_no_grants_is_incomplete_not_absence(self):
        v=SyntheticView()
        r=execute('{"action":"search","query":"встреча","question":""}',[v],set())
        self.assertEqual(r['status'],'INCOMPLETE')
        self.assertFalse(v.reads)
    def test_clarification_does_not_read(self):
        v=SyntheticView()
        r=execute('{"action":"clarify","query":"","question":"Какой документ?"}',[v],set())
        self.assertEqual(r['status'],'CLARIFY')
        self.assertFalse(v.reads)
    def test_malformed_and_unavailable_operations_rejected(self):
        for raw in ['[]','null','{"action":"delete","query":"x","question":""}',
                    '{"action":"search","query":"x","question":"","grants":[]}',
                    '{"action":"search","action":"clarify","query":"x","question":""}',
                    '{"action":"search","query":1,"question":""}',
                    '{"action":"search","query":"","question":""}']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):parse(raw)
    def test_utf8_budget(self):
        with self.assertRaises(ValueError):parse(json.dumps(dict(action='search',query='я'*129,question='')))
    def test_corruption_propagates_without_partial_answer(self):
        v=SyntheticView(corrupt=True)
        with self.assertRaises(ValueError):execute('{"action":"search","query":"встреча","question":""}',[v],{(v.version,'note.txt')})
    def test_unknown_grant_rejected(self):
        v=SyntheticView()
        with self.assertRaises(Exception):execute('{"action":"search","query":"x","question":""}',[v],{(v.version,'absent')})
    def test_remote_and_credential_urls_rejected_before_request(self):
        for url in ['https://example.com','http://192.168.1.1','http://localhost','http://127.0.0.1@evil.com','http://127.0.0.1/redirect']:
            with self.subTest(url=url),self.assertRaises(ValueError):local_plan('найди',endpoint=url)

if __name__=='__main__':unittest.main()
