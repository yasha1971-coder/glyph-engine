#!/usr/bin/env python3

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools" / "glyph_capability_probe.py"


class TestCapabilityProbeV1(unittest.TestCase):

    def test_probe_outputs_required_contract_fields(self):
        out = subprocess.run(
            [str(PROBE)],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout

        obj = json.loads(out)

        required = [
            "cpu_arch",
            "runtime_os",
            "runtime_kernel",
            "compiler_family",
            "compiler_version",
            "simd_capability",
            "query_protocol_version",
            "server_protocol_version",
            "http_protocol_version",
            "retrieval_contract_version",
            "capability_contract_version",
        ]

        for k in required:
            self.assertIn(k, obj)

        self.assertEqual(
            obj["query_protocol_version"],
            "GLYPH_QUERY_PROTOCOL_V1",
        )

        self.assertEqual(
            obj["retrieval_contract_version"],
            "GLYPH_RETRIEVAL_CONTRACT_V1",
        )

        self.assertEqual(
            obj["capability_contract_version"],
            "GLYPH_CAPABILITY_CONTRACT_V1",
        )

        self.assertIsInstance(obj["simd_capability"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
