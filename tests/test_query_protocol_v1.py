#!/usr/bin/env python3
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERY_BIN = ROOT / "build" / "query_fm_v1"


class TestQueryProtocolV1(unittest.TestCase):
    def test_json_output_is_deterministic_and_valid(self):
        fm = ROOT / "examples" / "mini" / "out" / "fm.bin"
        bwt = ROOT / "examples" / "mini" / "out" / "bwt.bin"
        pattern_hex = "6572726f72"

        out1 = subprocess.run(
            [str(QUERY_BIN), str(fm), str(bwt), pattern_hex, "--json"],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout

        out2 = subprocess.run(
            [str(QUERY_BIN), str(fm), str(bwt), pattern_hex, "--json"],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout

        self.assertEqual(out1, out2)

        obj = json.loads(out1)
        self.assertEqual(obj["pattern_hex"], pattern_hex)
        self.assertEqual(obj["interval"], [20, 22])
        self.assertEqual(obj["count"], 2)
        self.assertEqual(obj["fm_version"], "FMBINv2")
        self.assertEqual(obj["verified"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
