"""Executable product-gap probe, synthetic documents only; no LLM invoked.

Run directly for a JSON report. Passing tests validate measurement integrity,
not successful natural-language retrieval. Product gaps remain visible.
"""
import json
import contextlib
import io
import unittest

import test_local_memory_bridge as fixtures
from indexed_memory import ContentIndex


class PersonalRecallTests(unittest.TestCase):
    setUp = fixtures.LocalMemoryBridgeTests.setUp
    build = fixtures.LocalMemoryBridgeTests.build

    def measure(self):
        (self.source / 'note.txt').unlink()
        (self.source / 'private.txt').unlink()
        # Dates are fixture content, not invented filesystem timestamps.
        path = 'item-001.txt'
        old = '2006-04-12. Проект Маяк. Ирина. Доставка поездом. Согласовано 18 дней.'
        new = '2026-04-12. Проект Маяк. Ирина. Доставка самолётом. Согласовано 3 дня.'
        (self.source / path).write_text(old, encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            first = self.build('old')
        (self.source / path).write_text(new, encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            second = self.build('new')
        views = [first, second]
        grants = {(v.version, path) for v in views}
        cases = [
            ('literal_old', '18 дней', {first.version}, 'known phrase'),
            ('literal_new', '3 дня', {second.version}, 'known phrase'),
            ('shared_project', 'Маяк', {first.version, second.version}, 'both versions'),
            ('forgotten_name', 'Как мы собирались везти груз до перехода на авиацию?',
             {first.version}, 'paraphrase and temporal relation'),
            ('case_variant', 'маяк', {first.version, second.version}, 'letter case'),
            ('cross_language', 'доставка потягом', {first.version}, 'Ukrainian paraphrase'),
        ]
        rows = []
        idx = ContentIndex(views, grants)
        try:
            for ident, query, wanted, reason in cases:
                result = idx.query(query, grants)
                snippets = result['snippets']
                found = {s['version'] for s in snippets}
                for s in snippets:
                    view = next(v for v in views if v.version == s['version'])
                    data = view.read(s['path'])
                    self.assertEqual(data[s['byte_offset']:s['byte_offset'] + s['byte_length']], query.encode())
                rows.append({'id': ident, 'query': query, 'need': reason,
                             'expected_versions': len(wanted), 'returned_versions': len(found),
                             'backend_status': result['status'],
                             'outcome': 'MET' if found == wanted else 'GAP'})
            # A failed phrase is not evidence the relevant document is absent.
            for ident in ('forgotten_name', 'case_variant', 'cross_language'):
                row = next(r for r in rows if r['id'] == ident)
                self.assertEqual(row['outcome'], 'GAP')
                self.assertEqual(row['backend_status'], 'NO_MATCH_IN_INDEXED_SCOPE')
            self.assertEqual([r['outcome'] for r in rows[:3]], ['MET'] * 3)
            self.assertEqual(first.read(path), old.encode())
            self.assertEqual(second.read(path), new.encode())
        finally:
            idx.close()
        return {'format': 'GLYPH_PERSONAL_RECALL_PROBE_V1',
                'synthetic_only': True, 'llm_invoked': False,
                'scope': 'direct exact-index queries; not end-to-end assistant evaluation',
                'twenty_year_retention_tested': False,
                'cases': rows,
                'unmeasured': ['OCR', 'audio', 'event links', 'persistent index',
                               'real LLM query planning', 'GUI', 'competitor comparison']}

    def test_report_distinguishes_product_gaps_from_literal_controls(self):
        self.measure()


if __name__ == '__main__':
    case = PersonalRecallTests()
    case.setUp()
    try:
        print(json.dumps(case.measure(), ensure_ascii=False, indent=2))
    finally:
        case.doCleanups()
