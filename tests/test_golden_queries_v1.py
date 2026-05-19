#!/usr/bin/env python3
import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUERY_BIN = ROOT / "build" / "query_fm_v1"

FIXTURE = ROOT / "tests" / "fixtures" / "golden_queries_v1.json"


class TestGoldenQueriesV1(unittest.TestCase):
    def test_golden_queries(self):
        fixture = json.loads(FIXTURE.read_text())

        fm = ROOT / "examples" / "mini" / "out" / "fm.bin"
        bwt = ROOT / "examples" / "mini" / "out" / "bwt.bin"

        for q in fixture["queries"]:
            out = subprocess.run(
                [
                    str(QUERY_BIN),
                    str(fm),
                    str(bwt),
                    q["pattern_hex"],
                ],
                cwd=str(ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ).stdout

            m_interval = re.search(r"interval: \[(\d+), (\d+)\)", out)
            self.assertIsNotNone(m_interval)

            got_l = int(m_interval.group(1))
            got_r = int(m_interval.group(2))

            self.assertEqual(
                [got_l, got_r],
                q["expected_interval"]
            )

            m_count = re.search(r"count:\s+(\d+)", out)
            self.assertIsNotNone(m_count)

            got_count = int(m_count.group(1))

            self.assertEqual(
                got_count,
                q["expected_count"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
