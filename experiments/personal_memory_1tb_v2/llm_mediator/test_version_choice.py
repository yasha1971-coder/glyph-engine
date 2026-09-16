import json
import unittest
from version_choice import grade, request_body, DOCS


class VersionChoiceTests(unittest.TestCase):
    def test_literal_is_not_correct_version(self):
        r = grade(json.dumps(dict(action='select', id='v-c', quote='18 дней', question='')), DOCS, 'select', 'v-a')
        self.assertTrue(r['quote_is_literal'])
        self.assertFalse(r['decision_matches_oracle'])

    def test_fabrication_schema_and_abstention(self):
        for raw in ['[]', '{"action":"select","action":"select"}',
                    json.dumps(dict(action='select', id='v-a', quote='самолёт', question=''))]:
            self.assertFalse(grade(raw, DOCS, 'select', 'v-a')['valid'])
        r = grade(json.dumps(dict(action='clarify', id='', quote='', question='Какой проект?')), DOCS, 'clarify', None)
        self.assertTrue(r['decision_matches_oracle'])

    def test_request_has_no_oracle(self):
        payload = request_body('Найди документ', DOCS, 'local')
        self.assertEqual(json.loads(payload['messages'][1]['content']), {'question':'Найди документ', 'candidates':DOCS})


if __name__ == '__main__': unittest.main()
