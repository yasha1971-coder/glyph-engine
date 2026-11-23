import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
TOOL = HERE / "corpus_truth_gate.py"
SPEC = importlib.util.spec_from_file_location("truth_gate", TOOL)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class CorpusTruthGateTests(unittest.TestCase):
    def run_gate(self, source, state, *extra):
        return subprocess.run(
            [sys.executable, str(TOOL), "--source", str(source), "--state", str(state), *extra],
            text=True, capture_output=True,
        )

    def test_classification(self):
        self.assertEqual(gate.classify("photo.JPG"), ("image", ".jpg"))
        self.assertEqual(gate.classify("unknown"), ("other", ""))

    def test_inventory_does_not_follow_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, state, outside = root / "source", root / "state", root / "outside"
            source.mkdir(); outside.mkdir()
            (source / "one.txt").write_bytes(b"abc")
            (outside / "secret.bin").write_bytes(b"outside")
            os.symlink(outside, source / "link")
            result = self.run_gate(source, state)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json").read_text())
            self.assertEqual(receipt["inventory"]["by_kind"]["file"]["entries"], 1)
            self.assertEqual(receipt["inventory"]["by_kind"]["symlink"]["entries"], 1)
            self.assertFalse(receipt["source_content_read"])

    def test_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, state = root / "source", root / "state"
            (source / "a").mkdir(parents=True)
            (source / "a" / "x.bin").write_bytes(b"x")
            first = self.run_gate(source, state, "--max-directories", "1")
            self.assertEqual(first.returncode, gate.INCOMPLETE)
            second = self.run_gate(source, state)
            self.assertEqual(second.returncode, 0, second.stderr)
            receipt = json.loads((state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json").read_text())
            self.assertTrue(receipt["complete"])

    def test_state_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            result = self.run_gate(source, source / "state")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be disjoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
