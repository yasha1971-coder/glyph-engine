import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_settings as ab


def response(reason='stop'):
    return {'choices': [{'finish_reason': reason, 'message': {'content': json.dumps(
        {'action': 'not_found', 'id': '', 'quote': '', 'question': ''})}}]}


class ComparisonTests(unittest.TestCase):
    def test_schedule_balanced(self):
        jobs = list(ab.schedule(3))
        self.assertEqual(len(jobs), 60)
        self.assertEqual(sum(j[3] == 'greedy' for j in jobs), 30)
        self.assertNotEqual(jobs[0][3], jobs[2][3])

    def test_no_oracle_or_prompt_changes(self):
        for profile in ab.PROFILES:
            body = ab.make_body('Question', ab.probe.DOCS, 'model', profile, 1)
            self.assertEqual(body['messages'], ab.probe.request_body('Question', ab.probe.DOCS, 'model')['messages'])
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])

    def test_incomplete_never_accepted(self):
        result = ab.assess(response('length'), ab.probe.DOCS, 'not_found', None)
        self.assertTrue(result['decision_matches_oracle'])
        self.assertFalse(result['accepted_correct'])

    def test_wrong_literal_still_wrong(self):
        obj = response()
        obj['choices'][0]['message']['content'] = json.dumps(dict(action='select', id='v-b', quote='Поезд отменён.', question=''))
        self.assertFalse(ab.assess(obj, ab.probe.DOCS, 'select', 'v-a')['accepted_correct'])

    def test_remote_rejected(self):
        for endpoint in ['http://example.com', 'http://10.0.0.1', 'https://127.0.0.1',
                         'http://x@127.0.0.1', 'http://127.0.0.1/path']:
            with self.assertRaises(ValueError):
                ab.origin(endpoint)
        self.assertEqual(ab.origin('http://[::1]:8080'), ('::1', 8080))

    def test_fail_fast_and_saved_error(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.jsonl'
            with patch('builtins.print'):
                self.assertFalse(ab.run('http://127.0.0.1', 'model', output,
                                        caller=lambda *a: (_ for _ in ()).throw(TimeoutError())))
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[1]['error_type'], 'TimeoutError')
            self.assertFalse(records[-1]['complete'])
            with self.assertRaises(FileExistsError):
                ab.run('http://127.0.0.1', 'model', output)

    def test_full_run_and_no_false_perfect_score(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.jsonl'
            with patch('builtins.print'):
                self.assertTrue(ab.run('http://127.0.0.1', 'model', output, caller=lambda *a: response()))
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 22)
            for totals in records[-1]['results'].values():
                self.assertEqual(totals['correct'], 2)
                self.assertEqual(totals['attempted'], 10)


if __name__ == '__main__':
    unittest.main()
