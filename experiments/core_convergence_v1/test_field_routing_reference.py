"""Fail-closed composition+field routing research semantics, not runtime verification."""
import unittest
from dataclasses import replace

from field_routing_reference import (
    Rejected, canonical, digest, execute, fixture_documents, fixture_layouts,
    independently_verify, make_mask, make_pin, make_snapshot,
)


class FieldRoutingSemantics(unittest.TestCase):
    def setUp(self):
        self.docs = fixture_documents()
        self.a = make_snapshot(self.docs, fixture_layouts()[0])
        self.b = make_snapshot(self.docs, fixture_layouts()[1])
        self.mask = make_mask(self.a, "public")
        self.pin = make_pin(self.a, self.mask)
        self.available = tuple(x.runtime_index_id for x in self.a.blocks)

    def result(self, query=b"ana", limit=-1):
        return execute(self.a, self.mask, self.pin, self.available, query, limit)

    def prove(self, query=b"ana", limit=-1):
        r = self.result(query, limit)
        self.assertTrue(independently_verify(self.a, self.mask, self.pin,
                                             self.available, query, r, limit))
        return r

    def resign(self, result):
        r = dict(result)
        r.pop("result_id", None)
        r["result_id"] = digest(b"GLYPH_FIELD_MASK_FIXTURE_RESULT_V0\x00"
                                + canonical(r))
        return r

    def test_layouts_are_three_nonempty_contiguous_blocks(self):
        for s in (self.a, self.b):
            self.assertEqual(len(s.blocks), 3)
            self.assertEqual(s.blocks[0].start, 0)
            self.assertEqual(s.blocks[-1].end, 5)
            self.assertTrue(all(b.start < b.end for b in s.blocks))

    def test_logical_and_source_identity_partition_invariant(self):
        self.assertEqual(self.a.corpus_id, self.b.corpus_id)
        self.assertEqual(self.a.source_manifest_id, self.b.source_manifest_id)
        self.assertNotEqual(self.a.root_id, self.b.root_id)

    def test_field_mask_logical_id_invariant_to_partition(self):
        self.assertEqual(make_mask(self.a, "public").mask_id,
                         make_mask(self.b, "public").mask_id)

    def test_repartition_preserves_canonical_coordinates(self):
        for query in (b"ana", b"\x00a", b"absent", b"\nana", b"na"):
            a = self.prove(query)
            m = make_mask(self.b, "public")
            pin = make_pin(self.b, m)
            avail = tuple(b.runtime_index_id for b in self.b.blocks)
            b = execute(self.b, m, pin, avail, query)
            self.assertTrue(independently_verify(self.b, m, pin, avail, query, b))
            self.assertEqual((a["match_count"], a["coordinates"], a["status"]),
                             (b["match_count"], b["coordinates"], b["status"]))

    def test_public_count_and_overlapping_hits(self):
        r = self.prove()
        self.assertEqual(r["match_count"], 4)
        self.assertEqual(r["coordinates"], [[0, 0], [0, 5], [0, 7], [4, 4]])

    def test_private_duplicate_has_distinct_document_identity(self):
        mask = make_mask(self.a, "private")
        pin = make_pin(self.a, mask)
        r = execute(self.a, mask, pin, self.available, b"ana")
        self.assertEqual(r["coordinates"], [[3, 0], [3, 5], [3, 7]])
        self.assertTrue(independently_verify(self.a, mask, pin, self.available,
                                             b"ana", r))

    def test_duplicate_document_retained_in_corpus_identity(self):
        self.assertEqual(self.docs[0].data, self.docs[3].data)
        self.assertNotEqual(0, 3)
        changed = self.docs[:3] + self.docs[4:]
        s = make_snapshot(changed, ((0, 2), (2, 3), (3, 4)))
        self.assertNotEqual(s.corpus_id, self.a.corpus_id)

    def test_empty_document_keeps_global_position(self):
        self.assertEqual(self.docs[1].data, b"")
        self.assertEqual(self.prove()["coordinates"][-1][0], 4)

    def test_binary_zero_bytes_supported_in_semantic_fixture(self):
        self.assertEqual(self.prove(b"\x00a")["coordinates"], [[4, 0]])
        self.assertEqual(self.prove(b"\xffana")["coordinates"], [[4, 2]])

    def test_cross_document_match_is_not_a_hit(self):
        # Global concatenation of documents 2 and 3 contains b"\nana".
        self.assertIn(b"\nana", self.docs[2].data + self.docs[3].data)
        private = make_mask(self.a, "private")
        pin = make_pin(self.a, private)
        r = execute(self.a, private, pin, self.available, b"\nana")
        self.assertEqual(r["status"], "PROVEN_EMPTY")
        self.assertTrue(independently_verify(self.a, private, pin, self.available,
                                             b"\nana", r))

    def test_absence_only_under_complete_coverage(self):
        r = self.prove(b"not-a-present-substring")
        self.assertEqual(r["status"], "PROVEN_EMPTY")
        self.assertEqual(r["match_count"], 0)

    def test_zero_offset_limit_is_found_not_empty(self):
        r = self.prove(b"ana", 0)
        self.assertEqual(r["status"], "FOUND")
        self.assertEqual(r["match_count"], 4)
        self.assertEqual(r["coordinates"], [])
        self.assertFalse(r["offsets_complete"])

    def test_global_bounded_prefix(self):
        self.assertEqual(self.prove(b"ana", 2)["coordinates"], [[0, 0], [0, 5]])

    def test_completion_order_does_not_change_result(self):
        r = execute(self.a, self.mask, self.pin,
                    tuple(reversed(self.available)), b"ana")
        self.assertEqual(r, self.result())
        self.assertTrue(independently_verify(self.a, self.mask, self.pin,
                                             self.available, b"ana", r))

    def test_absent_middle_block_rejected_even_for_absent_pattern(self):
        with self.assertRaises(Rejected):
            execute(self.a, self.mask, self.pin, self.available[:1]
                    + self.available[2:], b"absent")

    def test_duplicate_available_block_rejected(self):
        with self.assertRaises(Rejected):
            execute(self.a, self.mask, self.pin,
                    (self.available[0],) * 3, b"ana")

    def test_extra_available_block_rejected(self):
        with self.assertRaises(Rejected):
            execute(self.a, self.mask, self.pin,
                    self.available + ("0" * 64,), b"ana")

    def test_missing_root_block_rejected(self):
        s = replace(self.a, blocks=self.a.blocks[:2])
        with self.assertRaises(Rejected):
            execute(s, self.mask, self.pin, self.available[:2], b"ana")

    def test_overlapping_layout_rejected(self):
        b = replace(self.a.blocks[1], start=1)
        s = replace(self.a, blocks=(self.a.blocks[0], b, self.a.blocks[2]))
        with self.assertRaises(Rejected):
            execute(s, self.mask, self.pin, self.available, b"ana")

    def test_empty_block_rejected(self):
        b = replace(self.a.blocks[1], end=self.a.blocks[1].start)
        s = replace(self.a, blocks=(self.a.blocks[0], b, self.a.blocks[2]))
        with self.assertRaises(Rejected):
            execute(s, self.mask, self.pin, self.available, b"ana")

    def test_corrupt_block_index_rejected(self):
        b = replace(self.a.blocks[0], runtime_index_id="0" * 64)
        s = replace(self.a, blocks=(b,) + self.a.blocks[1:])
        with self.assertRaises(Rejected):
            execute(s, self.mask, self.pin, self.available, b"ana")

    def test_corrupt_block_manifest_rejected(self):
        b = replace(self.a.blocks[0], runtime_manifest_sha256="0" * 64)
        s = replace(self.a, blocks=(b,) + self.a.blocks[1:])
        with self.assertRaises(Rejected):
            execute(s, self.mask, self.pin, self.available, b"ana")

    def test_modified_document_bytes_rejected(self):
        docs = (replace(self.docs[0], data=b"BAD"),) + self.docs[1:]
        with self.assertRaises(Rejected):
            execute(replace(self.a, documents=docs), self.mask,
                    self.pin, self.available, b"ana")

    def test_stale_external_pin_from_other_partition_rejected(self):
        mask = make_mask(self.b, "public")
        avail = tuple(b.runtime_index_id for b in self.b.blocks)
        with self.assertRaises(Rejected):
            execute(self.b, mask, self.pin, avail, b"ana")

    def test_external_pin_required(self):
        with self.assertRaises(Rejected):
            execute(self.a, self.mask, None, self.available, b"ana")

    def test_stale_generation_mask_rejected(self):
        newer = make_snapshot(self.docs, fixture_layouts()[0], "gen-2")
        mask = make_mask(newer, "public")
        with self.assertRaises(Rejected):
            execute(self.a, mask, self.pin, self.available, b"ana")

    def test_unpinned_private_mask_rejected(self):
        with self.assertRaises(Rejected):
            execute(self.a, make_mask(self.a, "private"), self.pin,
                    self.available, b"ana")

    def test_altered_mask_bit_rejected_even_with_old_hash(self):
        m = replace(self.mask, bits=(0,) + self.mask.bits[1:])
        with self.assertRaises(Rejected):
            execute(self.a, m, self.pin, self.available, b"absent")

    def test_stale_field_metadata_rejected(self):
        docs = self.docs[:2] + (replace(self.docs[2], label="public"),) + self.docs[3:]
        s = replace(self.a, documents=docs)
        with self.assertRaises(Rejected):
            execute(s, self.mask, self.pin, self.available, b"ana")

    def test_changed_result_coordinates_rejected(self):
        r = self.result()
        r["coordinates"][0][1] = 1
        with self.assertRaises(Rejected):
            independently_verify(self.a, self.mask, self.pin,
                                 self.available, b"ana", self.resign(r))

    def test_fake_zero_count_rejected(self):
        r = self.result()
        r["match_count"] = 0
        r["coordinates"] = []
        r["returned_count"] = 0
        r["status"] = "PROVEN_EMPTY"
        with self.assertRaises(Rejected):
            independently_verify(self.a, self.mask, self.pin,
                                 self.available, b"ana", self.resign(r))

    def test_wrong_query_receipt_rejected(self):
        with self.assertRaises(Rejected):
            independently_verify(self.a, self.mask, self.pin,
                                 self.available, b"banana", self.result())

    def test_changed_max_offsets_rejected_even_if_resigned(self):
        r = self.result(b"ana", 2)
        with self.assertRaises(Rejected):
            independently_verify(self.a, self.mask, self.pin,
                                 self.available, b"ana", self.resign(r), -1)

    def test_empty_query_rejected(self):
        with self.assertRaises(Rejected):
            self.result(b"")

    def test_invalid_path_and_duplicate_path_rejected(self):
        for docs in ((replace(self.docs[0], path="../bad"),) + self.docs[1:],
                     (replace(self.docs[0], path=self.docs[1].path),) + self.docs[1:]):
            s = make_snapshot(docs, fixture_layouts()[0])
            m = make_mask(s, "public")
            p = make_pin(s, m)
            with self.assertRaises(Rejected):
                execute(s, m, p, tuple(b.runtime_index_id for b in s.blocks), b"ana")


if __name__ == "__main__":
    unittest.main()
