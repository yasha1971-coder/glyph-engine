#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_INDEX = ROOT / "tools" / "build_glyph_index_v1.sh"
QUERY_VERIFIED = ROOT / "tools" / "query_verified_v1.py"


def build_index(corpus: bytes):
    td = tempfile.TemporaryDirectory()
    base = Path(td.name)
    raw = base / "corpus.bin"
    out = base / "index"
    raw.write_bytes(corpus)

    subprocess.run(
        [str(BUILD_INDEX), str(raw), str(out)],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return td, raw, out


def run_query(index_dir: Path, pattern: str):
    return subprocess.run(
        ["python3", str(QUERY_VERIFIED), str(index_dir), pattern],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class TestQueryVerified(unittest.TestCase):

    def test_verified_query_passes_on_valid_index(self):
        td, _raw, out = build_index(b"error alpha error beta")
        try:
            r = run_query(out, "error")
            self.assertEqual(r.returncode, 0, f"stdout={r.stdout}\nstderr={r.stderr}")
            self.assertIn("GLYPH INTEGRITY OK", r.stdout)
            self.assertIn("count:    2", r.stdout)
        finally:
            td.cleanup()

    def test_verified_query_fails_on_tampered_raw_corpus(self):
        td, _raw, out = build_index(b"error alpha error beta")
        try:
            manifest = json.loads((out / "manifest.json").read_text())
            raw_path = Path(manifest["raw_corpus"]["path"])
            raw_path.write_bytes(b"ERROR alpha ERROR beta")

            r = run_query(out, "error")
            self.assertEqual(r.returncode, 1, f"stdout={r.stdout}\nstderr={r.stderr}")
            self.assertIn("GLYPH INTEGRITY FAIL", r.stderr)
        finally:
            td.cleanup()

    def test_verified_query_fails_on_missing_artifact(self):
        td, _raw, out = build_index(b"error alpha error beta")
        try:
            (out / "fm.bin").unlink()

            r = run_query(out, "error")
            self.assertEqual(r.returncode, 1, f"stdout={r.stdout}\nstderr={r.stderr}")
            self.assertIn("GLYPH INTEGRITY FAIL", r.stderr)
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
