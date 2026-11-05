"""Executable synthetic bridge to the *existing* RLB3X proof-carrying V0 path.

This test does not create a new evidence format. It uses the existing
rlb3x_fixed.py, loc2_experimental.py, proof_carrying_recall_v0.py and
verify_proof_carrying_recall_v0.py on an intentionally tiny corpus.

The suffix array is a stdlib reference SA, not a compiled-SA performance test.
No real user files, external downloads or existing Vault directories are read.
"""
import copy
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "experiments" / "personal_vault_v0"
PY = sys.executable
PATTERN = "Wonderland"
DOCS = [
    ("doc-0.txt", b"Wonderland Wonderland\n"),
    ("doc-1.bin", b"other material \x00 and bytes \xff\n"),
]
BWT_HEADER_SIZE = struct.calcsize("<8sIQQIIIQQ")
SA_HEADER_SIZE = 60


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def run(*arguments, ok=True):
    p = subprocess.run(
        [PY, *map(str, arguments)], text=True, capture_output=True,
        timeout=30,
    )
    if ok and p.returncode:
        raise AssertionError(f"command exit {p.returncode}: {p.stderr[-1200:]}")
    return p


def emit_real_tiny_vault(root):
    vault = root / "vault"
    seg = vault / "segments" / "00000001"
    roots = vault / "manifests" / "roots"
    seg.mkdir(parents=True)
    roots.mkdir(parents=True)
    work = root / "builder-tmp"
    work.mkdir()
    raw = b"".join(blob for _, blob in DOCS)
    symbols = list(raw) + [256]
    # GLYPH virtual sentinel sorts before every real byte.
    key = lambda i: tuple(-1 if x == 256 else x for x in symbols[i:])
    sa = sorted(range(len(symbols)), key=key)
    bwt = [symbols[(i-1) % len(symbols)] for i in sa]
    (work / "sa.bin").write_bytes(b"\0" * SA_HEADER_SIZE + struct.pack("<" + "I" * len(sa), *sa))
    (work / "bwt.bin").write_bytes(b"\0" * BWT_HEADER_SIZE + struct.pack("<" + "H" * len(bwt), *bwt))
    run(V0 / "rlb3x_fixed.py", work / "bwt.bin",
        seg / "bwt.rlb3x", work / "builder-report.json", "--block-runs", 8)
    run(V0 / "loc2_experimental.py", "build", work / "sa.bin",
        len(symbols), 3, seg / "locate.loc2")
    run(V0 / "loc2_experimental.py", "verify", work / "sa.bin",
        seg / "locate.loc2")
    objects = []
    offset = 0
    for i, (name, blob) in enumerate(DOCS):
        objects.append({"id": i, "path": name, "offset": offset,
                        "bytes": len(blob), "sha256": sha(blob)})
        offset += len(blob)
    (seg / "objects.json").write_bytes(canonical({
        "format": "GLYPH_PERSONAL_VAULT_V0_OBJECT_MAP",
        "corpus_bytes": len(raw),
        "objects": objects,
    }))
    (seg / "source-state.json").write_bytes(canonical({
        "format": "GLYPH_VAULT_SOURCE_STATE_V0",
        "source_root": "synthetic-removed",
        "objects": [{"id": o["id"], "path": o["path"],
                     "bytes": o["bytes"], "sha256": o["sha256"],
                     "source_mtime_ns": 0} for o in objects],
    }))
    manifest = {
        "format": "GLYPH_VAULT_SEGMENT_MANIFEST_V0",
        "segment_id": "00000001",
        "created_unix_ns": 0,
        "source_root": "synthetic-removed",
        "object_count": len(objects),
        "source_bytes": len(raw),
        "files": {},
        "restore_tested": True,
        "object_hashes_verified": True,
        "eligible_to_free_source": False,
    }
    for key, filename in (
        ("rlb3x", "bwt.rlb3x"),
        ("loc2", "locate.loc2"),
        ("objects", "objects.json"),
        ("source_state", "source-state.json"),
    ):
        data = (seg / filename).read_bytes()
        manifest["files"][key] = {"name": filename, "bytes": len(data), "sha256": sha(data)}
    manifest_bytes = canonical(manifest)
    (seg / "segment-manifest.json").write_bytes(manifest_bytes)
    root_doc = {
        "format": "GLYPH_VAULT_ROOT_MANIFEST_V0",
        "committed_unix_ns": 0,
        "segments": ["00000001"],
        "segment_entries": [{
            "segment_id": "00000001",
            "segment_manifest_sha256": sha(manifest_bytes),
            "object_count": len(objects),
            "source_bytes": len(raw),
        }],
        "parent_root_name": None, "parent_root_sha256": None,
    }
    (roots / "00000000000000000001.json").write_bytes(canonical(root_doc))
    receipt = root / "receipt.json"
    made = run(V0 / "proof_carrying_recall_v0.py", vault,
               "--pattern", PATTERN, "--out", receipt)
    report = json.loads(made.stdout)
    assert report["occurrences"] == 2 and report["path"] == "doc-0.txt", report
    shutil.rmtree(work)
    assert not work.exists()
    return vault, receipt


class ExistingProofBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = tempfile.TemporaryDirectory(prefix="glyph-v0-proof-fixture-")
        cls.base = Path(cls.original.name)
        cls.base_vault, cls.base_receipt = emit_real_tiny_vault(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.original.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="glyph-v0-proof-mut-")
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        shutil.copytree(self.base_vault, self.vault)
        self.receipt = self.root / "receipt.json"
        shutil.copyfile(self.base_receipt, self.receipt)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, *, ok):
        p = run(V0 / "verify_proof_carrying_recall_v0.py",
                self.vault, self.receipt, ok=False)
        if ok:
            self.assertEqual(p.returncode, 0, p.stderr[-1600:])
            data = json.loads(p.stdout)
            self.assertTrue(data["ok"])
            self.assertTrue(data["verified_from_committed_glyph_state"])
            self.assertEqual(data["occurrences"], 2)
            self.assertFalse(data["original_source_required"])
        else:
            self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("VERIFY_FAIL", p.stdout + p.stderr)

    def mutate_receipt(self, update):
        data = json.loads(self.receipt.read_text())
        update(data)
        self.receipt.write_bytes(canonical(data))

    def test_positive_actual_rlb3x_restore_and_replay(self):
        self.assertFalse((self.base / "builder-tmp").exists())
        self.verify(ok=True)

    def test_generated_receipt_not_an_absence_proof(self):
        p = run(V0 / "proof_carrying_recall_v0.py", self.vault,
                "--pattern", "not-a-present-pattern-XYZ", "--out",
                self.root / "absent.json", ok=False)
        self.assertNotEqual(p.returncode, 0)

    def test_cross_document_match_rejected(self):
        # This exact byte string occurs only across the two concatenated objects.
        p = run(V0 / "proof_carrying_recall_v0.py", self.vault,
                "--pattern", "\nother", "--out",
                self.root / "boundary.json", ok=False)
        self.assertNotEqual(p.returncode, 0)

    def test_wrong_root_hash(self):
        self.mutate_receipt(lambda r: r["vault_commit"].update(root_sha256="0"*64))
        self.verify(ok=False)

    def test_wrong_manifest_hash(self):
        self.mutate_receipt(lambda r: r["segment"].update(segment_manifest_sha256="0"*64))
        self.verify(ok=False)

    def test_wrong_object_sha(self):
        self.mutate_receipt(lambda r: r["object"].update(sha256="0"*64))
        self.verify(ok=False)

    def test_wrong_object_offset(self):
        self.mutate_receipt(lambda r: r["object"].update(segment_corpus_offset=1))
        self.verify(ok=False)

    def test_wrong_claimed_occurrence(self):
        self.mutate_receipt(lambda r: r["evidence"]["object_offsets"].__setitem__(0, 1))
        self.verify(ok=False)

    def test_missing_occurrence_rejected(self):
        self.mutate_receipt(lambda r: (
            r["evidence"]["object_offsets"].pop(),
            r["evidence"].update(occurrences=1),
        ))
        self.verify(ok=False)

    def test_inconsistent_receipt_count(self):
        self.mutate_receipt(lambda r: r["evidence"].update(occurrences=7))
        self.verify(ok=False)

    def test_wrong_query_binding(self):
        self.mutate_receipt(lambda r: r["claim"].update(pattern_hex=b"impostor".hex()))
        self.verify(ok=False)

    def test_mutated_compressed_payload(self):
        p = self.vault / "segments" / "00000001" / "bwt.rlb3x"
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(data)
        self.verify(ok=False)

    def test_mutated_locate_index(self):
        p = self.vault / "segments" / "00000001" / "locate.loc2"
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(data)
        self.verify(ok=False)

    def test_mutated_object_map(self):
        p = self.vault / "segments" / "00000001" / "objects.json"
        data = bytearray(p.read_bytes())
        data[-2] ^= 1
        p.write_bytes(data)
        self.verify(ok=False)

    def test_mutated_root(self):
        p = next((self.vault / "manifests" / "roots").glob("*.json"))
        p.write_bytes(p.read_bytes() + b" ")
        self.verify(ok=False)

    def test_mutated_segment_manifest(self):
        p = self.vault / "segments" / "00000001" / "segment-manifest.json"
        p.write_bytes(p.read_bytes() + b" ")
        self.verify(ok=False)


if __name__ == "__main__":
    unittest.main()
