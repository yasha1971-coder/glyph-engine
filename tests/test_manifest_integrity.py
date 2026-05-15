# tests/test_manifest_integrity.py
#
# Manifest integrity regression suite.
# Tests real GLYPH_INDEX_MANIFEST_V1 format.
# stdlib only. No server. No HTTP.
#
# Invariants:
#   M1 — missing manifest       → fail(1)
#   M2 — wrong format field     → fail(1)
#   M3 — raw_corpus size change → fail(1)
#   M4 — raw_corpus tampered    → fail(1)
#   M5 — sentinel corpus tampered → fail(1)
#   M6 — wrong sentinel value   → fail(1)
#   M7 — missing artifact       → fail(1)
#   M8 — valid state            → pass(0)

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.verify_manifest_v1 import verify


# ── test corpus ───────────────────────────────────────────────────────────────

RAW_CORPUS    = b"the cat sat on the mat the cat"          # no 0x00
INDEX_CORPUS  = RAW_CORPUS + b"\x00"                       # appended sentinel


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_valid_index(d: str) -> Path:
    """
    Write a complete valid index under directory d.
    Returns Path(d).
    """
    index_dir = Path(d)

    # corpus files
    raw_path      = index_dir / "corpus.bin"
    sentinel_path = index_dir / "corpus.sentinel.bin"
    raw_path.write_bytes(RAW_CORPUS)
    sentinel_path.write_bytes(INDEX_CORPUS)

    # artifact stubs (content irrelevant for manifest checks)
    for name in ("sa.bin", "bwt.bin", "fm.bin"):
        (index_dir / name).write_bytes(b"\x00STUB")

    manifest = {
        "format": "GLYPH_INDEX_MANIFEST_V1",
        "raw_corpus": {
            "path": str(raw_path),
            "bytes": len(RAW_CORPUS),
            "sha256": sha256_bytes(RAW_CORPUS),
        },
        "index_corpus": {
            "path": str(sentinel_path),
            "bytes": len(INDEX_CORPUS),
            "sha256": sha256_bytes(INDEX_CORPUS),
            "sentinel": "0x00",
        },
        "artifacts": {
            "sa":  str(index_dir / "sa.bin"),
            "bwt": str(index_dir / "bwt.bin"),
            "fm":  str(index_dir / "fm.bin"),
        },
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return index_dir


# ─────────────────────────────────────────────────────────────────────────────

class TestManifestIntegrity(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _index_dir(self) -> Path:
        return build_valid_index(self._tmp)

    def _manifest(self) -> dict:
        p = Path(self._tmp) / "manifest.json"
        return json.loads(p.read_text())

    def _write_manifest(self, m: dict):
        p = Path(self._tmp) / "manifest.json"
        p.write_text(json.dumps(m, indent=2))

    # ── M8: valid state ───────────────────────────────────────────────────────

    def test_M8_valid_passes(self):
        index_dir = self._index_dir()
        try:
            verify(index_dir)
        except SystemExit as e:
            self.fail(f"verify() exited {e.code} on valid index")

    # ── M1: manifest missing ──────────────────────────────────────────────────

    def test_M1_missing_manifest(self):
        index_dir = self._index_dir()
        (index_dir / "manifest.json").unlink()
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    # ── M2: wrong format field ────────────────────────────────────────────────

    def test_M2_wrong_format(self):
        index_dir = self._index_dir()
        m = self._manifest()
        m["format"] = "GLYPH_INDEX_MANIFEST_V0"   # wrong version
        self._write_manifest(m)
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    # ── M3: raw_corpus size changed ───────────────────────────────────────────

    def test_M3_raw_corpus_size_mismatch(self):
        index_dir = self._index_dir()
        # Append bytes to corpus after index was built
        (index_dir / "corpus.bin").write_bytes(RAW_CORPUS + b"EXTRA")
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    # ── M4: raw_corpus tampered (same size, different bytes) ──────────────────

    def test_M4_raw_corpus_tampered(self):
        index_dir = self._index_dir()
        tampered = b"X" * len(RAW_CORPUS)   # same size, wrong content
        (index_dir / "corpus.bin").write_bytes(tampered)
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    # ── M5: sentinel corpus tampered ──────────────────────────────────────────

    def test_M5_sentinel_corpus_tampered(self):
        index_dir = self._index_dir()
        tampered = b"Y" * len(INDEX_CORPUS)
        (index_dir / "corpus.sentinel.bin").write_bytes(tampered)
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    # ── M6: wrong sentinel value in manifest ──────────────────────────────────

    def test_M6_wrong_sentinel_value(self):
        index_dir = self._index_dir()
        m = self._manifest()
        m["index_corpus"]["sentinel"] = "0x01"   # wrong sentinel
        self._write_manifest(m)
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    # ── M7: artifact missing ──────────────────────────────────────────────────

    def test_M7_missing_artifact_fm(self):
        index_dir = self._index_dir()
        (index_dir / "fm.bin").unlink()
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)

    def test_M7_missing_artifact_sa(self):
        index_dir = self._index_dir()
        (index_dir / "sa.bin").unlink()
        with self.assertRaises(SystemExit) as ctx:
            verify(index_dir)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
