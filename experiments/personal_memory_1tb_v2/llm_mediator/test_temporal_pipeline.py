import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import temporal_pipeline as t


class PipelineTests(unittest.TestCase):
    def test_category_separate_from_anchor(self):
        d, valid = t.parse_intent('{"temporal":"latest","anchor":"invented"}', 'Последнее решение')
        self.assertEqual(d['temporal'], 'latest')
        self.assertFalse(valid)

    def test_duplicate_and_unknown_rejected(self):
        for raw in ['{"temporal":"latest","temporal":"first","anchor":""}',
                    '{"temporal":"invented","anchor":""}']:
            with self.assertRaises(ValueError):
                t.parse_intent(raw, 'question')

    def test_no_oracle_and_direct_unchanged(self):
        b = t.selection_body('q', t.probe.DOCS, 'm')
        self.assertEqual(b['messages'], t.probe.request_body('q', t.probe.DOCS, 'm')['messages'])
        b = t.selection_body('q', t.probe.DOCS, 'm', {'temporal':'unspecified', 'anchor':''})
        self.assertEqual(set(json.loads(b['messages'][1]['content'])), {'question','candidates','temporal_hint'})

    def test_false_abstention_is_not_success(self):
        def caller(endpoint, method, path, body, timeout):
            if body['messages'][0]['content'] == t.INTENT_PROMPT:
                answer = {'temporal':'unspecified','anchor':''}
            else:
                answer = {'action':'not_found','id':'','quote':'','question':''}
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'r.jsonl'
            with patch('builtins.print'):
                self.assertTrue(t.run('http://127.0.0.1', 'm', p, caller))
            rs = [json.loads(l) for l in p.read_text().splitlines()]
            self.assertEqual(sum(r['type']=='call' for r in rs),30)
            for result in rs[-1]['results'].values():
                self.assertEqual(result['correct'],2)
                self.assertEqual(result['attempted'],10)
            with self.assertRaises(FileExistsError):
                t.run('http://127.0.0.1','m',p,caller)

    def test_timeout_stops_and_saves(self):
        def fail(*args):
            raise TimeoutError()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'r.jsonl'
            with patch('builtins.print'):
                self.assertFalse(t.run('http://127.0.0.1','m',p,fail))
            rs=[json.loads(l) for l in p.read_text().splitlines()]
            self.assertEqual(sum(r['type']=='call' for r in rs),1)
            self.assertFalse(rs[-1]['complete'])

if __name__ == '__main__':
    unittest.main()
