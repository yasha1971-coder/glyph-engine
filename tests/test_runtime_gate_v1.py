#!/usr/bin/env python3

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestRuntimeGateV1(unittest.TestCase):

    def test_runtime_gate_passes(self):

        proc = subprocess.run(
            [
                "python3",
                str(ROOT / "tools" / "glyph_runtime_gate.py")
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        obj = json.loads(proc.stdout)

        self.assertEqual(
            obj["runtime_gate_version"],
            "GLYPH_RUNTIME_GATE_V1",
        )

        self.assertTrue(
            obj["ready"]
        )

        self.assertEqual(
            obj["capability_contract"],
            "PASS",
        )

        self.assertEqual(
            obj["retrieval_contract"],
            "PASS",
        )

        self.assertEqual(
            obj["manifest_integrity"],
            "PASS",
        )

        self.assertEqual(
            obj["fm_artifact"],
            "PASS",
        )

def test_runtime_gate_fails_on_missing_manifest(self):

        env = os.environ.copy()
        env["GLYPH_GATE_MANIFEST"] = "/tmp/glyph_missing_manifest_for_test.json"

        proc = subprocess.run(
            [
                "python3",
                str(ROOT / "tools" / "glyph_runtime_gate.py")
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        self.assertNotEqual(proc.returncode, 0)

        obj = json.loads(proc.stdout)

        self.assertFalse(obj["ready"])

        self.assertEqual(
            obj["manifest_integrity"],
            "FAIL",
        )

def test_runtime_gate_fails_on_missing_fm_artifact(self):

        env = os.environ.copy()
        env["GLYPH_GATE_FM"] = "/tmp/glyph_missing_fm_for_test.bin"

        proc = subprocess.run(
            [
                "python3",
                str(ROOT / "tools" / "glyph_runtime_gate.py")
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        self.assertNotEqual(proc.returncode, 0)

        obj = json.loads(proc.stdout)

        self.assertFalse(obj["ready"])

        self.assertEqual(
            obj["fm_artifact"],
            "FAIL",
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
