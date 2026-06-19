#!/usr/bin/env python3
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MINI_RUN = ROOT / "examples" / "mini" / "run_mini.sh"
MAKE_ARTIFACT = ROOT / "tools" / "glyph_make_audit_artifact_v0.py"
VERIFY_ARTIFACT = ROOT / "tools" / "glyph_verify_audit_artifact_v0.py"
MAKE_CASE = ROOT / "tools" / "glyph_make_evidence_case_v1.py"

INDEX_DIR = ROOT / "examples" / "mini" / "out"
ARTIFACT = INDEX_DIR / "audit_artifact_v0.json"
CASE = INDEX_DIR / "evidence_case_v1.json"


class TestEvidenceCaseV1(unittest.TestCase):

    def run_cmd(self, cmd):
        return subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_mini_evidence_case_v1(self):
        self.run_cmd([str(MINI_RUN)])

        make_artifact = self.run_cmd([
            "python3",
            str(MAKE_ARTIFACT),
            "--index-dir",
            "examples/mini/out",
            "--query",
            "error",
            "--output",
            "examples/mini/out/audit_artifact_v0.json",
        ])

        self.assertIn("reproduce_status=PASS", make_artifact.stdout)
        self.assertIn("match_count=2", make_artifact.stdout)
        self.assertIn("offset_mode=locate_backend_v2", make_artifact.stdout)
        self.assertIn("offsets=[0, 37]", make_artifact.stdout)

        verify_artifact = self.run_cmd([
            "python3",
            str(VERIFY_ARTIFACT),
            "examples/mini/out/audit_artifact_v0.json",
        ])

        self.assertIn("VERIFY AUDIT ARTIFACT OK", verify_artifact.stdout)

        make_case = self.run_cmd([
            "python3",
            str(MAKE_CASE),
            "--artifact",
            "examples/mini/out/audit_artifact_v0.json",
            "--output",
            "examples/mini/out/evidence_case_v1.json",
        ])

        self.assertIn("records=2", make_case.stdout)

        case = json.loads(CASE.read_text(encoding="utf-8"))

        self.assertEqual(case["case_version"], "GLYPH_EVIDENCE_CASE_V1")
        self.assertEqual(case["query"]["text"], "error")
        self.assertEqual(case["query"]["hex"], "6572726f72")
        self.assertEqual(case["result_summary"]["match_count"], 2)
        self.assertEqual(case["result_summary"]["fm_interval"], [20, 22])
        self.assertEqual(case["result_summary"]["offset_mode"], "locate_backend_v2")
        self.assertEqual(case["result_summary"]["offsets"], [0, 37])

        records = case["evidence_records"]
        self.assertEqual(len(records), 2)

        self.assertEqual(records[0]["offset"], 0)
        self.assertEqual(records[0]["match_text"], "error")
        self.assertTrue(records[0]["byte_check"])

        self.assertEqual(records[1]["offset"], 37)
        self.assertEqual(records[1]["match_text"], "error")
        self.assertTrue(records[1]["byte_check"])

        self.assertTrue(all(r["byte_check"] for r in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
