import sys
import tempfile
import unittest
import zlib
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verified_hybrid_archive as archive
import local_memory_bridge as bridge
from test_verified_hybrid_archive import make_inventory
import verified_reversible_precompression_archive as precomp
from test_verified_reversible_precompression_archive import make_precomp


class LocalMemoryBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "note.txt").write_text("Решение: сохранить оригинал.\n", encoding="utf-8")
        (self.source / "private.txt").write_text("секрет", encoding="utf-8")

    def build(self, name):
        inv, dest = self.root / (name + "-inv"), self.root / name
        make_inventory(self.source, inv)
        archive.build(inv, dest, 0)
        pin = archive.sha256_bytes((dest / archive.RECEIPT).read_bytes())
        return bridge.ArchiveView(dest, pin)

    def test_two_additions_rename_versions_restore_and_old_bytes_unchanged(self):
        first = self.build("first")
        before = {str(p.relative_to(first.root)): p.read_bytes() for p in first.root.rglob("*") if p.is_file()}
        (self.source / "note.txt").rename(self.source / "renamed.txt")
        (self.source / "note.txt").write_text("Новое решение: проверить копию.", encoding="utf-8")
        second = self.build("second")
        grants = {(v.version, p) for v in (first, second) for p in v.files}
        result = bridge.evidence([first, second], "оригинал", grants)
        self.assertEqual(result["matching_files"], 2)
        self.assertEqual({s["version"] for s in result["snippets"]}, {first.version, second.version})
        self.assertEqual(first.files["note.txt"]["sha256"], second.files["renamed.txt"]["sha256"])
        for v in (first, second):
            target = self.root / (v.root.name + "-restored")
            archive.restore(v.root, target)
            for path in v.files:
                self.assertEqual((target / path).read_bytes(), v.read(path))
        self.assertEqual(before, {str(p.relative_to(first.root)): p.read_bytes() for p in first.root.rglob("*") if p.is_file()})

    def test_private_paths_never_disclosed_without_grant(self):
        v = self.build("a")
        r = bridge.evidence([v], "секрет", {(v.version, "note.txt")})
        self.assertEqual(r["status"], "PROVEN_EMPTY")
        self.assertNotIn("private.txt", str(r))

    def test_empty_scope_is_not_proven_empty(self):
        self.assertEqual(bridge.evidence([], "x", set())["status"], "INCOMPLETE")

    def test_context_budget_preserves_utf8_and_byte_coordinates(self):
        v = self.build("a")
        r = bridge.evidence([v], "Решение", {(v.version, "note.txt")}, context_bytes=17)
        self.assertLessEqual(r["context_bytes"], 17)
        s = r["snippets"][0]
        self.assertEqual(v.read(s["path"])[s["byte_offset"]:s["byte_offset"] + s["byte_length"]], s["text"].encode())

    def test_small_context_still_reports_match(self):
        v = self.build("a")
        r = bridge.evidence([v], "Решение", {(v.version, "note.txt")}, context_bytes=1)
        self.assertEqual(r["status"], "FOUND")
        self.assertTrue(r["context_truncated"])
        self.assertEqual(r["snippets"], [])

    def test_binary_and_scan_budget_do_not_claim_absence(self):
        (self.source / "binary").write_bytes(b"\xff\x00")
        v = self.build("a")
        for path, budget in (("binary", 1024), ("note.txt", 1)):
            r = bridge.evidence([v], "missing", {(v.version, path)}, scan_bytes=budget)
            self.assertEqual(r["status"], "INCOMPLETE")

    def test_tampering_discards_evidence(self):
        v = self.build("a")
        obj = v.objects[v.files["private.txt"]["sha256"]]
        (v.root / obj["path"]).write_bytes(b"corrupt")
        with self.assertRaises(bridge.Error):
            bridge.evidence([v], "Решение", {(v.version, p) for p in v.files})

    def test_wrong_pin_and_unknown_grant_rejected(self):
        v = self.build("a")
        with self.assertRaises(bridge.Error):
            bridge.ArchiveView(v.root, "0" * 64)
        with self.assertRaises(bridge.Error):
            bridge.evidence([v], "x", {(v.version, "nonexistent")})

    def test_symlink_object_rejected(self):
        v = self.build("a")
        obj = v.objects[v.files["note.txt"]["sha256"]]
        p = v.root / obj["path"]
        outside = self.root / "outside"
        p.rename(outside)
        p.symlink_to(outside)
        with self.assertRaises(bridge.Error):
            v.read("note.txt")

    def test_external_codec_never_executed_by_query(self):
        v = self.build("a")
        v.objects[v.files["note.txt"]["sha256"]]["codec"] = "precomp-cn"
        r = bridge.evidence([v], "Решение", {(v.version, "note.txt")})
        self.assertEqual(r["status"], "INCOMPLETE")

    def test_real_precomp_manifest_mixed_coverage(self):
        (self.source / "photo.jpeg").write_bytes(b"FAKEJPEG" + random.Random(73).randbytes(10000))
        binary, digest = make_precomp(self.root)
        inv, dest = self.root / "inv", self.root / "precomp-archive"
        make_inventory(self.source, inv)
        precomp.build(inv, dest, 0, binary, digest)
        pin = archive.sha256_bytes((dest / precomp.RECEIPT).read_bytes())
        v = bridge.ArchiveView(dest, pin)
        r = bridge.evidence([v], "Решение", {(v.version, p) for p in v.files})
        self.assertEqual(r["status"], "FOUND")
        self.assertFalse(r["coverage_complete"])
        self.assertEqual([x["path"] for x in r["skipped"]], ["photo.jpeg"])

    def test_oversized_decoding_rejected(self):
        v = self.build("a")
        v.limit = 1024
        obj = v.objects[v.files["note.txt"]["sha256"]]
        payload = zlib.compress(b"X" * 100000)
        (v.root / obj["path"]).write_bytes(payload)
        obj.update(codec="deflate9", stored_bytes=len(payload), stored_sha256=archive.sha256_bytes(payload))
        with self.assertRaises(bridge.Error):
            v.read("note.txt")


if __name__ == "__main__":
    unittest.main()
