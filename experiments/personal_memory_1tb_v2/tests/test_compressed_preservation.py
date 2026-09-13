import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import compressed_preservation as backend
import verified_hybrid_archive as base
import verified_reversible_precompression_archive as precomp
from test_verified_hybrid_archive import make_inventory
from test_verified_reversible_precompression_archive import make_precomp

spec = importlib.util.spec_from_file_location("runtime_integration", HERE.parents[1] / "deploy/glyph-v1/portable_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
PROFILE = json.loads((HERE.parents[1] / "deploy/glyph-v1/PROFILE.json").read_text())


class CompressedPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / "source"
        (source / "family").mkdir(parents=True)
        self.data = b"FAKEJPEG" + random.Random(4).randbytes(4096)
        (source / "family/photo.jpeg").write_bytes(self.data)
        self.digest = hashlib.sha256(self.data).hexdigest()
        inv = self.root / "inventory"
        make_inventory(source, inv)
        self.binary, binary_pin = make_precomp(self.root)
        archive = self.root / "archive"
        precomp.build(inv, archive, 0, self.binary, binary_pin)
        self.pin = base.sha256_bytes((archive / precomp.RECEIPT).read_bytes())
        self.backend = backend.CompressedPreservation(archive, self.pin, self.binary, binary_pin)

    def test_selected_precomp_object_exact_bytes(self):
        self.assertEqual(self.backend.read_verified("family/photo.jpeg", len(self.data), self.digest), self.data)

    def test_wrong_identity_refused_before_decoder(self):
        with patch.object(precomp, "decode", side_effect=AssertionError("must not decode")):
            with self.assertRaises(base.ArchiveError):
                self.backend.read_verified("family/photo.jpeg", len(self.data), "0" * 64)

    def test_changed_binary_refused(self):
        self.binary.write_bytes(b"changed")
        with self.assertRaises(base.ArchiveError):
            self.backend.read_verified("family/photo.jpeg", len(self.data), self.digest)

    def host_case(self, invalid_selection=False, bad_backend=False):
        for folder in ("vault", "trust"):
            (self.root / folder).mkdir()
        phase = self.root / "phase.json"
        phase.write_text(json.dumps({
            "release_manifest_sha256": PROFILE["frozen_files"]["frozen/RELEASE_MANIFEST.json"],
            "frozen_resolver_sha256": PROFILE["frozen_files"]["frozen/hybrid_resolver_v613.py"],
            "target_oracle_present": False, "query_specific_bridge_added": False,
            "human_selection_required": True, "payload_touched": False, "materialized": False,
            "shortlist": [{"name": "family", "parent": "family"}]}))
        selection = self.root / "selection.json"
        runtime.create_selection(phase, 1, selection)
        if invalid_selection:
            phase.write_text(phase.read_text() + "\n")
        args = SimpleNamespace(vault=str(self.root / "vault"), trust_root=str(self.root / "trust"),
                               phase_a=str(phase), selection=str(selection), output=str(self.root / "output"),
                               pin=None, pin_sha256=None)
        obj = {"path": "family/photo.jpeg", "bytes": len(self.data), "sha256": self.digest, "offset": 0}
        hard = SimpleNamespace(sha=lambda p: "reader", authenticate_root_chain=lambda *a: ({}, None),
                               authenticate_segment=lambda *a: ({"objects": [obj]}, None, None, None, None))
        pin = {"query_reader_sha256": "reader", "segment_ids": ["01"], "segments": [{"segment_id": "01"}]}
        # Authentication fixtures are stubbed here: this tests host sequencing and
        # backend integration, NOT a new proof of V1 authentication correctness.
        with patch.object(runtime, "canonical_pin", return_value=(None, pin)), \
             patch.object(runtime, "relocated_pin", return_value=pin), \
             patch.object(runtime, "load_module", return_value=hard), \
             patch.object(self.backend, "read_verified", wraps=self.backend.read_verified) as read:
            if bad_backend:
                read.side_effect = lambda *a: b"wrong"
            if invalid_selection or bad_backend:
                with self.assertRaises(runtime.Refusal):
                    runtime.run_materialization(args, self.backend)
                if invalid_selection:
                    read.assert_not_called()
                self.assertEqual(list((self.root / "output").glob("materialized-*")), [])
            else:
                runtime.run_materialization(args, self.backend)
                files = list((self.root / "output").glob("materialized-*/*"))
                self.assertEqual(len(files), 1)
                self.assertEqual(files[0].read_bytes(), self.data)
                report = json.loads(next((self.root / "output").glob("GLYPH_V2_COMPRESSED_PHASE_B_*.json")).read_text())
                self.assertEqual(report["preservation"]["receipt_sha256"], self.pin)

    def test_existing_host_materializes_without_cache(self):
        self.host_case()

    def test_changed_selection_binding_never_decodes(self):
        self.host_case(invalid_selection=True)

    def test_host_rechecks_backend_before_publication(self):
        self.host_case(bad_backend=True)
