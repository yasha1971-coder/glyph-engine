import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("glyph_v1_portable_runtime", HERE / "portable_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class PortableRuntimeTests(unittest.TestCase):
    def test_frozen_and_evidence_identities(self):
        result = runtime.verify_profile()
        self.assertEqual(result["status"], "GREEN")
        self.assertEqual(result["frozen_files_verified"], 5)
        self.assertEqual(result["evidence_files_verified"], 6)

    def test_pin_relocation_preserves_identity_fields(self):
        _, pin = runtime.canonical_pin()
        moved = runtime.relocated_pin(pin, "/srv/vault", "/srv/trust", "/opt/query.py")
        self.assertEqual(moved["root_sha256"], pin["root_sha256"])
        self.assertEqual(moved["segments"][0]["rank_sha256"], pin["segments"][0]["rank_sha256"])
        self.assertEqual(moved["segments"][0]["rlb_path"], "/srv/vault/segments/00000001/bwt.rlb3x")
        self.assertEqual(moved["segments"][0]["rank_path"], "/srv/trust/segments/00000001/rank.s1.auth")

    def test_human_selection_is_bound_to_phase_a_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase = root / "phase-a.json"
            phase.write_text(
                json.dumps(
                    {
                        "release_manifest_sha256": "de3161726e336127647af4596ee5e7c0588965bb1cc21b687276d42259b0e36b",
                        "frozen_resolver_sha256": "cefc8611ed96a490d978efc851611ca7ed4f1ab7690be0ace72687a3b5e0e6d9",
                        "target_oracle_present": False,
                        "query_specific_bridge_added": False,
                        "human_selection_required": True,
                        "payload_touched": False,
                        "materialized": False,
                        "shortlist": [{"name": "candidate", "parent": "candidate"}],
                    }
                ),
                encoding="utf-8",
            )
            selection = root / "selection.json"
            receipt = runtime.create_selection(phase, 1, selection)
            self.assertEqual(receipt["candidate_number"], 1)
            self.assertEqual(receipt["phase_a_sha256"], runtime.sha256_file(phase))
            self.assertFalse(receipt["payload_touched"])

    def test_object_slice_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpus.bin"
            corpus.write_bytes(b"prefix-object-suffix")
            data = b"object"
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(runtime.verified_slice(corpus, 7, 6, expected), data)
            with self.assertRaises(runtime.Refusal):
                runtime.verified_slice(corpus, 7, 6, "0" * 64)

    def test_custom_pin_requires_external_identity(self):
        pin = HERE / "frozen" / "readpath-pin-v2.json"
        with self.assertRaises(runtime.Refusal):
            runtime.canonical_pin(pin)
        _, loaded = runtime.canonical_pin(pin, runtime.sha256_file(pin))
        self.assertEqual(loaded["format"], "GLYPH_LAPTOP_AUTH_READPATH_PIN_V2_MULTI")

    def test_phase_a_reproduction_has_oracle_boundary(self):
        frozen = (HERE / "frozen" / "hybrid_resolver_v613.py").read_text(encoding="utf-8")
        runner = (HERE / "reproduction" / "phase_a_history.py").read_text(encoding="utf-8")
        self.assertEqual(frozen.count("for case in CASES:\n"), 1)
        self.assertIn("# Oracle.\n", frozen)
        self.assertIn('"target_oracle_present":False', runner)
        self.assertIn('"payload_touched":False', runner)


if __name__ == "__main__":
    unittest.main()
