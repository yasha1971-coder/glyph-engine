import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import selection_ablation as a


class AblationTests(unittest.TestCase):
    def test_only_intent_differs_between_treatments(self):
        case, question, _, _ = a.probe.CASES[0]
        plain = a.body_for('instruction_only', case, question, a.probe.DOCS, 'm')
        oracle = a.body_for('oracle_intent', case, question, a.probe.DOCS, 'm')
        self.assertEqual(plain['messages'][0], oracle['messages'][0])
        data = json.loads(oracle['messages'][1]['content'])
        hint = data.pop('temporal_hint')
        self.assertEqual(set(hint), {'temporal', 'anchor'})
        self.assertEqual(data, json.loads(plain['messages'][1]['content']))

    def test_baseline_unchanged(self):
        body = a.body_for('direct', 'old', 'q', a.probe.DOCS, 'm')
        self.assertEqual(body['messages'], a.probe.request_body('q', a.probe.DOCS, 'm')['messages'])

    def test_schedule_and_oracle_anchors(self):
        jobs = list(a.jobs())
        self.assertEqual(len(jobs), 30)
        for mode in a.MODES:
            self.assertEqual(sum(j[2] == mode for j in jobs), 10)
        for case, question, _, _ in a.probe.CASES:
            hint = a.ORACLE_HINTS[case]
            self.assertTrue(a.pipeline.parse_intent(json.dumps(hint), question)[1])

    def test_complete_mock_run_not_perfect(self):
        def caller(*args):
            return {'choices': [{'finish_reason': 'stop', 'message': {'content':
                json.dumps(dict(action='not_found', id='', quote='', question=''))}}]}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'r.jsonl'
            with patch('builtins.print'):
                self.assertTrue(a.run('http://127.0.0.1', 'm', p, caller))
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            for mode, result in rows[-1]['results'].items():
                self.assertEqual(result['correct'], 2)
                self.assertEqual(result['uses_oracle_intent'], mode == 'oracle_intent')
            with self.assertRaises(FileExistsError):
                a.run('http://127.0.0.1','m',p,caller)

    def test_truncated_response_stops_without_success(self):
        def caller(*args):
            return {'choices': [{'finish_reason':'length','message': {'content':
                json.dumps(dict(action='select',id='v-a',quote='Ирина согласовала поезд.',question=''))}}]}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'r.jsonl'
            with patch('builtins.print'):
                self.assertFalse(a.run('http://127.0.0.1','m',p,caller))
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]['results']['direct']['correct'], 0)

if __name__ == '__main__':
    unittest.main()
