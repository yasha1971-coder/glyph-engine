import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from authenticated_fixture import make_fixture
from test_compressed_preservation import runtime, PROFILE, backend, base, make_inventory


class RealAuthenticatedPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = {"family/note.txt": "решение original\n".encode(), "other/data.bin": b"\x00\xffbinary"}
        self.vault, self.trust, self.pin = make_fixture(self.root, runtime.FROZEN, self.files)
        source = self.root / "source"
        for name, data in self.files.items():
            p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        inv, archive = self.root / "inv", self.root / "archive"
        make_inventory(source, inv)
        with contextlib.redirect_stdout(io.StringIO()):
            base.build(inv, archive, 0)
        self.backend = backend.CompressedPreservation(archive, base.sha256_bytes((archive / base.RECEIPT).read_bytes()))
        phase = self.root / "phase.json"
        # Phase A is an explicit fixture, not a claim of natural-language discovery.
        phase.write_text(json.dumps({"release_manifest_sha256": PROFILE["frozen_files"]["frozen/RELEASE_MANIFEST.json"],
            "frozen_resolver_sha256": PROFILE["frozen_files"]["frozen/hybrid_resolver_v613.py"],
            "human_selection_required": True, "payload_touched": False, "materialized": False,
            "target_oracle_present": False, "query_specific_bridge_added": False,
            "shortlist": [{"name": "family", "parent": "family"}]}))
        selection = self.root / "selection.json"
        runtime.create_selection(phase, 1, selection)
        self.args = SimpleNamespace(vault=str(self.vault), trust_root=str(self.trust), pin=str(self.pin),
            pin_sha256=runtime.sha256_file(self.pin), phase_a=str(phase), selection=str(selection), output=str(self.root / "out"))

    def search(self, query):
        cp = subprocess.run([sys.executable, str(runtime.FROZEN / "glyph-trust2-multi.py"), "search",
                             str(self.vault), str(self.pin), query], capture_output=True, text=True)
        return cp.returncode, json.loads(cp.stdout)

    def test_real_fm_locate_matches_bytes_and_absence(self):
        rc, r = self.search("original")
        self.assertEqual(rc, 0)
        self.assertEqual(r["state"], "FOUND")
        self.assertEqual(r["valid_count"], 1)
        self.assertEqual(r["hits"][0]["object_offset"], self.files["family/note.txt"].index(b"original"))
        self.assertEqual(self.search("not present")[1]["state"], "PROVEN_EMPTY")

    def test_real_host_authentication_and_compressed_materialization(self):
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.run_materialization(self.args, self.backend)
        outputs = list((self.root / "out").glob("materialized-*/*"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].read_bytes(), self.files["family/note.txt"])

    def test_corrupt_object_map_blocks_materialization(self):
        p = self.vault / "segments/01/objects.json"
        p.write_bytes(p.read_bytes().replace(b"binary", b"Binary") + b" ")
        with self.assertRaisesRegex(Exception, "objects size failed"):
            runtime.run_materialization(self.args, self.backend)
        self.assertEqual(list((self.root / "out").glob("materialized-*")), [])

    def test_corrupt_root_blocks_materialization(self):
        p = self.vault / "manifests/roots/1.json"
        p.write_text(p.read_text() + " ")
        with self.assertRaisesRegex(Exception, "root authentication failed"):
            runtime.run_materialization(self.args, self.backend)

    def test_corrupt_selected_compressed_object_is_not_published(self):
        obj = self.backend.view.objects[self.backend.view.files["family/note.txt"]["sha256"]]
        p = self.backend.root / obj["path"]
        raw = bytearray(p.read_bytes())
        raw[0] ^= 1
        p.write_bytes(raw)
        with self.assertRaisesRegex(base.ArchiveError, "stored bytes corrupted"):
            runtime.run_materialization(self.args, self.backend)
        self.assertEqual(list((self.root / "out").glob("materialized-*")), [])

    def test_touched_index_corruption_is_untrusted(self):
        p = self.vault / "segments/01/bwt.rlb3x"
        raw = bytearray(p.read_bytes())
        raw[-8] ^= 1
        p.write_bytes(raw)
        rc, result = self.search("original")
        self.assertEqual(rc, 3)
        self.assertEqual(result["state"], "UNTRUSTED")

    def test_changed_external_pin_rejected(self):
        self.pin.write_text(self.pin.read_text() + "\n")
        with self.assertRaises(runtime.Refusal):
            runtime.run_materialization(self.args, self.backend)


if __name__ == "__main__":
    unittest.main()
