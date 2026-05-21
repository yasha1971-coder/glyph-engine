#!/usr/bin/env python3

import json
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
