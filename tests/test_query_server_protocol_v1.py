#!/usr/bin/env python3
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_BIN = ROOT / "build" / "query_fm_server_v1"


class TestQueryServerProtocolV1(unittest.TestCase):
    def test_server_json_protocol(self):
        fm = ROOT / "examples" / "mini" / "out" / "fm.bin"
        bwt = ROOT / "examples" / "mini" / "out" / "bwt.bin"

        proc = subprocess.Popen(
            [str(SERVER_BIN), str(fm), str(bwt), "--json"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        ready = proc.stderr.readline().strip()
        self.assertEqual(ready, "READY")

        proc.stdin.write("6572726f72\n")
        proc.stdin.write("__EXIT__\n")
        proc.stdin.flush()

        line = proc.stdout.readline().strip()
        proc.wait(timeout=5)

        obj = json.loads(line)
        self.assertEqual(obj["pattern_hex"], "6572726f72")
        self.assertEqual(obj["interval"], [20, 22])
        self.assertEqual(obj["count"], 2)
        self.assertEqual(obj["fm_version"], "FMBINv2")
        self.assertEqual(obj["verified"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
