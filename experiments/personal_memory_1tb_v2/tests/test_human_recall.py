"""Research-informed synthetic recall cases, not observed participant queries."""
import contextlib
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import test_local_memory_bridge as fixtures
from indexed_memory import ContentIndex, Error


class HumanRecallTests(unittest.TestCase):
    setUp = fixtures.LocalMemoryBridgeTests.setUp
    build = fixtures.LocalMemoryBridgeTests.build

    def corpus(self):
        (self.source / 'note.txt').unlink()
        (self.source / 'private.txt').unlink()
        docs = {
            'a.txt': '2006. Проект Маяк. Ирина согласовала доставку поездом за 18 дней.',
            'b.txt': '2026. Проект Маяк. Доставка поездом отменена. Самолётом за 3 дня.',
            'c.txt': '2020. Отель Маяк. Бронирование на 18 дней. Оплачено 840 евро.',
            'd.txt': '2019. Гарантия на холодильник. Чек на 840 евро. Werkstraße 7.',
            'e.txt': '2021. Café. Файл cafe\u0301 имеет другой способ записи акцента.',
            'private.txt': 'Закрытый документ: Ирина, Маяк, холодильник.',
        }
        for name, text in docs.items():
            (self.source / name).write_text(text, encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            v = self.build('human')
        grants = {(v.version, p) for p in docs if p != 'private.txt'}
        return v, grants

    def test_human_case_report(self):
        v, grants = self.corpus()
        cases = [
            ('case', 'маяк', {'a.txt', 'b.txt', 'c.txt'}, 'candidate retrieval'),
            ('german', 'WERKSTRASSE', {'d.txt'}, 'candidate retrieval'),
            ('amount_ambiguous', '840 евро', {'c.txt', 'd.txt'}, 'both candidates required'),
            ('old_phrase', 'поездом за 18 дней', {'a.txt'}, 'literal retrieval'),
            ('paraphrase', 'Где чек на кухонную технику?', {'d.txt'}, 'semantic relevance'),
            ('typo', 'холодилник', {'d.txt'}, 'typo tolerance'),
            ('language', 'доставка потягом', {'a.txt'}, 'cross-language relevance'),
            ('negation', 'Доставка поездом', {'a.txt'}, 'approved train plan, exclude cancellation'),
            ('absent', 'подводная лодка', set(), 'no literal match; not universal absence'),
        ]
        report = []
        for folded in (False, True):
            idx = ContentIndex([v], grants, casefold=folded)
            try:
                for ident, query, expected, criterion in cases:
                    result = idx.query(query, grants)
                    got = {s['path'] for s in result['snippets']}
                    for s in result['snippets']:
                        raw = v.read(s['path'])
                        self.assertEqual(raw[s['byte_offset']:s['byte_offset']+s['byte_length']], s['text'].encode())
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), s['sha256'])
                    report.append(dict(case=ident, query=query, mode='casefold' if folded else 'exact',
                                       criterion=criterion, outcome='MET' if got == expected else 'GAP',
                                       expected=sorted(expected), returned=sorted(got)))
            finally:
                idx.close()
        self.assertEqual(sum(r['outcome']=='MET' for r in report[:9]), 3)
        self.assertEqual(sum(r['outcome']=='MET' for r in report[9:]), 5)
        self.report = dict(synthetic=True, participant_study=False, llm_invoked=False,
                           statistical_accuracy_claim=False, results=report)

    def test_unicode_offsets_original_text_and_literals(self):
        (self.source / 'note.txt').write_text('🙂 Straße ИРИНА ﬃ Σίσυφος abc" OR xyz * % _', encoding='utf-8')
        v = self.build('unicode'); grants = {(v.version, 'note.txt')}
        idx = ContentIndex([v], grants, casefold=True)
        self.addCleanup(idx.close)
        for q in ('STRASSE', 'ирина', 'FFI', 'ΣΊΣΥΦΟΣ', 'abc" or xyz', '* % _'):
            s = idx.query(q, grants)['snippets'][0]
            self.assertIn(q.casefold(), s['text'].casefold())
            raw = v.read(s['path'])
            self.assertEqual(raw[s['byte_offset']:s['byte_offset']+s['byte_length']], s['text'].encode())

    def test_permission_corruption_and_revocation(self):
        v, grants = self.corpus()
        with patch.object(v, 'read', wraps=v.read) as read:
            idx = ContentIndex([v], grants, casefold=True)
            self.addCleanup(idx.close)
            self.assertNotIn('private.txt', [c.args[0] for c in read.call_args_list])
        with self.assertRaises(Error): idx.query('маяк', set())
        obj = v.objects[v.files['a.txt']['sha256']]
        (v.root / obj['path']).write_bytes(b'broken')
        with self.assertRaises(Error): idx.query('ирина', grants)


if __name__ == '__main__':
    case = HumanRecallTests()
    case.setUp()
    try:
        case.test_human_case_report()
        print(json.dumps(case.report, ensure_ascii=False, indent=2))
    finally:
        case.doCleanups()
