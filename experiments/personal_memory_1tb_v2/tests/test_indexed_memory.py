import unittest
from unittest.mock import patch

import test_local_memory_bridge as fixtures
import indexed_memory as index
from llm_mediator.mediator import execute_indexed


class IndexedMemoryTests(unittest.TestCase):
    setUp = fixtures.LocalMemoryBridgeTests.setUp
    build = fixtures.LocalMemoryBridgeTests.build

    def make(self, views, grants, **options):
        result = index.ContentIndex(views, grants, **options)
        self.addCleanup(result.close)
        return result

    def test_real_archive_exact_multilingual_and_operators(self):
        text = 'Решение: оригінал. Keep ORIGINAL. abc" OR xyz * % _ ; --'
        (self.source / 'note.txt').write_text(text)
        v = self.build('a'); grants = {(v.version, 'note.txt')}
        idx = self.make([v], grants)
        for q in ('Решение', 'оригінал', 'ORIGINAL', 'abc" OR xyz', '* % _', '; --'):
            r = idx.query(q, grants)
            self.assertEqual(r['status'], 'FOUND', q)
            s = r['snippets'][0]
            self.assertEqual(v.read(s['path'])[s['byte_offset']:s['byte_offset'] + s['byte_length']], q.encode())
        self.assertEqual(idx.query('original', grants)['status'], 'NO_MATCH_IN_INDEXED_SCOPE')

    def test_permission_filtered_before_reads_and_revocation(self):
        v = self.build('a'); grants = {(v.version, 'note.txt')}
        with patch.object(v, 'read', wraps=v.read) as read:
            idx = self.make([v], grants)
            self.assertEqual([c.args[0] for c in read.call_args_list], ['note.txt'])
        self.assertNotIn('private', str(idx.query('секрет', grants)))
        with self.assertRaises(index.Error): idx.query('Решение', set())

    def test_versions_do_not_collapse(self):
        a = self.build('a')
        (self.source / 'note.txt').write_text('Решение: другая версия.')
        b = self.build('b'); grants = {(v.version, 'note.txt') for v in (a,b)}
        idx = self.make([a,b], grants)
        self.assertEqual({s['version'] for s in idx.query('Решение', grants)['snippets']}, {a.version,b.version})
        self.assertEqual(idx.query('оригинал', grants)['snippets'][0]['version'], a.version)

    def test_incomplete_binary_budget_and_empty_scope(self):
        (self.source / 'binary').write_bytes(b'\xff\x00')
        v = self.build('a'); grants = {(v.version,p) for p in v.files}
        for options in ({}, {'source_budget':1}):
            idx = self.make([v], grants, **options)
            self.assertEqual(idx.query('missing', grants)['status'], 'INCOMPLETE')
        self.assertEqual(self.make([], set()).query('missing', set())['status'], 'INCOMPLETE')

    def test_corruption_after_index_never_returns_evidence(self):
        v = self.build('a'); grants = {(v.version, 'note.txt')}
        idx = self.make([v], grants)
        obj = v.objects[v.files['note.txt']['sha256']]
        (v.root / obj['path']).write_bytes(b'broken')
        with self.assertRaises(index.Error): idx.query('Решение', grants)

    def test_corruption_during_build_fails(self):
        v = self.build('a'); obj = v.objects[v.files['note.txt']['sha256']]
        (v.root / obj['path']).write_bytes(b'broken')
        with self.assertRaises(index.Error): index.ContentIndex([v], {(v.version,'note.txt')})

    def test_unsupported_codec_is_explicit(self):
        v = self.build('a'); grants = {(v.version,'note.txt')}
        v.objects[v.files['note.txt']['sha256']]['codec'] = 'precomp-cn'
        r = self.make([v], grants).query('Решение', grants)
        self.assertEqual(r['status'], 'INCOMPLETE')
        self.assertEqual(r['skipped'][0]['reason'], 'unsupported-codec-or-file-size')

    def test_limits_and_lifecycle(self):
        v = self.build('a'); grants = {(v.version,p) for p in v.files}
        idx = self.make([v], grants)
        for q in ('', 'ab', 'a\x00b', 'x'*257):
            with self.assertRaises(ValueError): idx.query(q, grants)
        with self.assertRaises(index.Error): self.make([v,v], grants)
        with self.assertRaises(index.Error): self.make([v], {(v.version,'missing')})
        idx.close()
        with self.assertRaises(index.Error): idx.query('Решение', grants)

    def test_result_limit_reports_truncation(self):
        (self.source / 'private.txt').write_text('Решение: второе.')
        v = self.build('a'); grants = {(v.version,p) for p in v.files}
        r = self.make([v], grants).query('Решение', grants, limit=1)
        self.assertTrue(r['truncated'])
        self.assertEqual(len(r['snippets']), 1)

    def test_planner_adapter_and_no_mutation_actions(self):
        v = self.build('a'); grants = {(v.version,'note.txt')}
        idx = self.make([v], grants)
        r = execute_indexed('{"action":"search","query":"Решение","question":""}', idx, grants)
        self.assertEqual(r['status'], 'FOUND')
        r = execute_indexed('{"action":"search","query":"ab","question":""}', idx, grants)
        self.assertEqual(r['status'], 'CLARIFY')
        with self.assertRaises(ValueError):
            execute_indexed('{"action":"delete","query":"note.txt","question":""}', idx, grants)


if __name__ == '__main__':
    unittest.main()
