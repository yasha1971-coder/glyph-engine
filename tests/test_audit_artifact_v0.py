#!/usr/bin/env python3
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MINI_RUN = ROOT / "examples" / "mini" / "run_mini.sh"
MAKE_ARTIFACT = ROOT / "tools" / "glyph_make_audit_artifact_v0.py"
VERIFY_ARTIFACT = ROOT / "tools" / "glyph_verify_audit_artifact_v0.py"

INDEX_DIR = ROOT / "examples" / "mini" / "out"
ARTIFACT = INDEX_DIR / "audit_artifact_v0.json"


class TestAuditArtifactV0(unittest.TestCase):

    def run_cmd(self, cmd):
        return subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_mini_audit_artifact_v0(self):
        self.run_cmd([str(MINI_RUN)])

        make = self.run_cmd([
            "python3",
            str(MAKE_ARTIFACT),
            "--index-dir",
            "examples/mini/out",
            "--query",
            "error",
            "--output",
            "examples/mini/out/audit_artifact_v0.json",
        ])

        self.assertIn("reproduce_status=PASS", make.stdout)
        self.assertIn("match_count=2", make.stdout)
        self.assertIn("offset_mode=locate_backend_v2", make.stdout)
        self.assertIn("offsets=[0, 37]", make.stdout)

        verify = self.run_cmd([
            "python3",
            str(VERIFY_ARTIFACT),
            "examples/mini/out/audit_artifact_v0.json",
        ])

        self.assertIn("VERIFY AUDIT ARTIFACT OK", verify.stdout)

        data = json.loads(ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(data["artifact_version"], "GLYPH_AUDIT_ARTIFACT_V0")
        self.assertEqual(data["query"]["hex"], "6572726f72")
        self.assertEqual(data["result"]["match_count"], 2)
        self.assertEqual(data["result"]["fm_interval"], [20, 22])
        self.assertEqual(data["result"]["offset_mode"], "locate_backend_v2")
        self.assertEqual(data["result"]["offsets"], [0, 37])
        self.assertEqual(data["verification"]["reproduce_status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
